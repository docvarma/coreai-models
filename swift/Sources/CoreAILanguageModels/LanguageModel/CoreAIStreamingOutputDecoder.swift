// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Strict decoder shared by the Core AI text and vision adapters.
///
/// Inline profiles (`plainChat`, `qwen35XML`, `gemma4Channels`) carry response
/// text at top level, so text is released as soon as it can no longer be part
/// of a pending structural marker. A tool block is the exception: it is
/// withheld until its closing marker arrives, because a structurally
/// incomplete call cannot be dispatched. Envelope profiles (`harmony`,
/// `atem`) carry no top-level text at all: every message is wrapped in a
/// `<|start|>assistant…<|message|>BODY<terminator>` envelope, so a header is
/// held whole, classified once complete, and its body then streams — or, for
/// a tool recipient, buffers — until the envelope terminates.
package struct CoreAIStreamingOutputDecoder {
    package enum Event: Equatable, Sendable {
        case response(String)
        case reasoning(String)
        case toolCall(id: String, name: String, argumentsJSON: String)
    }

    private enum Mode {
        /// Response text at top level, reasoning and tool blocks inline.
        case inline
        /// Whole output is a sequence of channel-message envelopes.
        case envelope
    }

    private enum State {
        case text
        case reasoning
        case toolBlock
    }

    private let profile: CoreAILanguageProtocolProfile
    private let reasoningEnabled: Bool
    private var scanner = MarkerScanner()
    private var state: State = .text
    private var blockBuffer = ""
    /// Load-bearing: a second reasoning block is rejected outright. That is the
    /// only reason a `trimLeadingReasoningNewline` left set by an empty
    /// reasoning block can never be applied to a later one.
    private var sawReasoning = false
    private var trimLeadingReasoningNewline = false
    private var toolCallIndex = 0
    private var envelopePhase: EnvelopePhase = .header
    private var sawResponseEnvelope = false
    private var sawToolEnvelope = false

    private var mode: Mode {
        switch profile {
        case .plainChat, .qwen35XML, .gemma4Channels: .inline
        case .harmony, .atem: .envelope
        }
    }

    package init(
        profile: CoreAILanguageProtocolProfile,
        reasoningEnabled: Bool
    ) {
        self.profile = profile
        self.reasoningEnabled = reasoningEnabled
    }

    package mutating func consume(_ delta: String) throws -> [Event] {
        scanner.append(delta)
        switch mode {
        case .inline: return try drainInline(isFinal: false)
        case .envelope: return try drainEnvelope(isFinal: false)
        }
    }

    package mutating func finish() throws -> [Event] {
        switch mode {
        case .inline: return try drainInline(isFinal: true)
        case .envelope: return try drainEnvelope(isFinal: true)
        }
    }

    // MARK: - Inline marker vocabulary

    /// Marker that opens a reasoning block. Deliberately *not* gated on the
    /// reasoning policy: a disabled model that reasons anyway must be caught,
    /// not streamed to the caller as response text.
    private var reasoningOpen: String? {
        switch profile {
        case .qwen35XML: "<think>"
        case .gemma4Channels: "<|channel>thought"
        case .plainChat, .harmony, .atem: nil
        }
    }

    private var reasoningClose: String? {
        switch profile {
        case .qwen35XML: "</think>"
        case .gemma4Channels: "<channel|>"
        case .plainChat, .harmony, .atem: nil
        }
    }

    /// For `qwen35XML` this is the *outer* pair; the body handed to the block
    /// parser begins with `<function=`, which `parseQwenXMLCall` reads itself.
    private var toolOpen: String? {
        switch profile {
        case .qwen35XML: "<tool_call>"
        case .gemma4Channels: "<|tool_call>call:"
        case .plainChat, .harmony, .atem: nil
        }
    }

    private var toolClose: String? {
        switch profile {
        case .qwen35XML: "</tool_call>"
        case .gemma4Channels: "<tool_call|>"
        case .plainChat, .harmony, .atem: nil
        }
    }

    /// Reserved markers that belong to this profile's protocol but never form
    /// a transition: looser spellings of a block delimiter, and — for
    /// `plainChat`, which has no protocol of its own — every reserved marker.
    /// Seeing one where text is being streamed is a protocol violation, never
    /// content, so each carries the failure it raises.
    private var strayProfileMarkers: [(marker: String, failure: CoreAIProtocolFailure)] {
        switch profile {
        case .plainChat:
            CoreAITranscriptCodec.reservedMarkers.map { ($0, .malformedChannel) }
        case .qwen35XML:
            []
        case .gemma4Channels:
            [("<|channel>", .malformedChannel), ("<|tool_call>", .malformedToolCall)]
        case .harmony, .atem:
            []
        }
    }

    /// Markers the scanner watches in the current state. In a state whose text
    /// reaches the caller this is the profile's whole vocabulary, not just the
    /// markers that advance the state machine: a delimiter in the wrong
    /// position must raise a typed failure rather than stream out as response
    /// or reasoning text. `transition(on:into:)` sorts legal from illegal.
    ///
    /// Inside a tool block only the closing delimiter matters — that text is
    /// buffered for the block parser and never reaches the caller, so a
    /// marker-like byte in a tool argument is the block parser's business.
    private var pendingMarkers: [String] {
        switch state {
        case .text, .reasoning:
            [reasoningOpen, reasoningClose, toolOpen, toolClose].compactMap { $0 }
                + strayProfileMarkers.map(\.marker)
        case .toolBlock:
            [toolClose].compactMap { $0 }
        }
    }

    // MARK: - Inline drain

    private mutating func drainInline(isFinal: Bool) throws -> [Event] {
        var events: [Event] = []
        while true {
            let markers = pendingMarkers
            if let match = scanner.firstSettledMatch(of: markers, isFinal: isFinal) {
                let before = scanner.takeUpTo(match.range)
                events += try emit(before)
                try transition(on: match.marker, into: &events)
                continue
            }
            let safe = scanner.takeSafe(waitingFor: markers, isFinal: isFinal)
            events += try emit(safe)
            if isFinal { try assertClosed() }
            return events
        }
    }

    private mutating func emit(_ text: String) throws -> [Event] {
        guard !text.isEmpty else { return [] }
        switch state {
        case .text:
            return [.response(text)]
        case .reasoning:
            var body = text
            if trimLeadingReasoningNewline {
                trimLeadingReasoningNewline = false
                if body.hasPrefix("\n") { body.removeFirst() }
            }
            return body.isEmpty ? [] : [.reasoning(body)]
        case .toolBlock:
            blockBuffer.append(text)
            return []
        }
    }

    private mutating func transition(on marker: String, into events: inout [Event]) throws {
        if marker == reasoningOpen {
            guard reasoningEnabled else { throw failure(.malformedReasoning) }
            guard state != .reasoning else { throw failure(.nestedProtocolBlock) }
            guard !sawReasoning else { throw failure(.duplicateProtocolBlock) }
            sawReasoning = true
            trimLeadingReasoningNewline = true
            state = .reasoning
        } else if marker == reasoningClose {
            guard state == .reasoning else {
                throw failure(sawReasoning ? .duplicateProtocolBlock : .malformedReasoning)
            }
            state = .text
        } else if marker == toolOpen {
            guard state == .text else { throw failure(.nestedProtocolBlock) }
            blockBuffer = ""
            state = .toolBlock
        } else if marker == toolClose {
            guard state == .toolBlock else { throw failure(.malformedToolCall) }
            events.append(try parseToolBlock(blockBuffer, index: toolCallIndex))
            toolCallIndex += 1
            blockBuffer = ""
            state = .text
        } else {
            // A reserved spelling that never forms a transition anywhere.
            throw failure(
                strayProfileMarkers.first { $0.marker == marker }?.failure ?? .malformedChannel)
        }
    }

    private func assertClosed() throws {
        switch state {
        case .text: return
        case .reasoning: throw failure(.unfinishedReasoning)
        case .toolBlock: throw failure(.unfinishedToolCall)
        }
    }

    /// One structurally complete tool block, without its delimiting markers.
    private func parseToolBlock(_ body: String, index: Int) throws -> Event {
        switch profile {
        case .qwen35XML: return try parseQwenXMLCall(body, callIndex: index)
        case .gemma4Channels: return try parseGemmaCall(body, callIndex: index)
        case .plainChat, .harmony, .atem: throw failure(.malformedToolCall)
        }
    }

    // MARK: - Envelope marker vocabulary

    /// Where the body of the envelope currently open is routed.
    private enum Destination: Equatable {
        case response
        case reasoning
        /// `harmony`: the body is the JSON argument object of one named call.
        case toolArguments(name: String)
        /// `atem`: the body is an `<atem:function_calls>` block that names its
        /// own calls, so the header recipient carries no usable name.
        case toolMarkup
    }

    private enum EnvelopePhase {
        case header
        case body(Destination)
    }

    /// Header terminator. Everything before it is protocol, never content.
    private static let envelopeHeaderEnd = "<|message|>"

    /// The terminators the batch parsers matched on, one set per profile.
    private var envelopeTerminators: [String] {
        switch profile {
        case .harmony: ["<|end|>", "<|return|>", "<|call|>"]
        case .atem: ["<|eom|>", "<|eot|>"]
        case .plainChat, .qwen35XML, .gemma4Channels: []
        }
    }

    /// One vocabulary per phase, driving both matching and hold-back so the
    /// two cannot drift — the discipline `pendingMarkers` applies inline.
    ///
    /// Inside a buffered tool body only the terminators matter: that text
    /// never reaches the caller, so marker-like bytes in tool arguments are
    /// the block parser's business.
    private var pendingEnvelopeMarkers: [String] {
        switch envelopePhase {
        case .header: [Self.envelopeHeaderEnd]
        case .body: envelopeTerminators
        }
    }

    // MARK: - Envelope drain

    private mutating func drainEnvelope(isFinal: Bool) throws -> [Event] {
        var events: [Event] = []
        while true {
            let markers = pendingEnvelopeMarkers
            switch envelopePhase {
            case .header:
                guard let match = scanner.firstSettledMatch(of: markers, isFinal: isFinal)
                else {
                    // A header is held whole; a partial one at the end of the
                    // stream is a truncated envelope, not content.
                    guard isFinal else { return events }
                    let remainder = scanner.takeAll()
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    if !remainder.isEmpty { throw failure(.malformedChannel) }
                    return events
                }
                let target = try classify(header: scanner.takeUpTo(match.range))
                try assertOrdered(target)
                envelopePhase = .body(target)
            case .body(let target):
                if let match = scanner.firstSettledMatch(of: markers, isFinal: isFinal) {
                    let body = scanner.takeUpTo(match.range)
                    events += try flushBody(body, to: target, terminatedBy: match.marker)
                    envelopePhase = .header
                    continue
                }
                let safe = scanner.takeSafe(waitingFor: markers, isFinal: isFinal)
                events += try flushBody(safe, to: target, terminatedBy: nil)
                if isFinal { throw failure(unfinished(target)) }
                return events
            }
        }
    }

    /// Classifies one complete envelope header, using the spellings the batch
    /// parsers matched on: harmony puts the recipient *before* the channel
    /// (`<|start|>assistant to=functions.NAME<|channel|>commentary`), atem puts
    /// it directly after the role (`<|start|>assistant to=user`). A recipient
    /// wins over a channel name. Anything else is a protocol violation rather
    /// than text, so it is never streamed to the caller.
    ///
    /// Note the pipes on both sides of harmony's `<|channel|>`: `gemma4Channels`
    /// spells its own channel marker `<|channel>`, and the two are different
    /// markers.
    private func classify(header rawHeader: String) throws -> Destination {
        // Whitespace between envelopes belongs to no message.
        let header = String(rawHeader.drop(while: { $0.isWhitespace }))
        let role = "<|start|>assistant"
        guard header.hasPrefix(role) else { throw failure(.malformedChannel) }
        let remainder = String(header.dropFirst(role.count))
        switch profile {
        case .harmony:
            if remainder == "<|channel|>analysis" { return try reasoningDestination() }
            if remainder == "<|channel|>final" { return .response }
            let recipient = " to=functions."
            if remainder.hasPrefix(recipient),
                let channel = remainder.range(of: "<|channel|>commentary")
            {
                let start = remainder.index(remainder.startIndex, offsetBy: recipient.count)
                let name = String(remainder[start..<channel.lowerBound])
                guard isValidName(name) else { throw failure(.malformedToolCall) }
                return .toolArguments(name: name)
            }
            throw failure(.malformedChannel)
        case .atem:
            let recipient = " to="
            guard remainder.hasPrefix(recipient) else { throw failure(.malformedChannel) }
            switch String(remainder.dropFirst(recipient.count)) {
            case "self": return try reasoningDestination()
            case "user": return .response
            default: return .toolMarkup
            }
        case .plainChat, .qwen35XML, .gemma4Channels:
            throw failure(.malformedChannel)
        }
    }

    /// Neither envelope profile can suppress reasoning in its template, so a
    /// reasoning envelope under a disabled policy is caught here rather than
    /// relabelled as response text.
    private func reasoningDestination() throws -> Destination {
        guard reasoningEnabled else { throw failure(.malformedReasoning) }
        return .reasoning
    }

    /// Ordering rules carried over from the batch parsers: nothing follows the
    /// response envelope, and reasoning never follows a tool call. Consecutive
    /// tool envelopes are legal.
    private func assertOrdered(_ destination: Destination) throws {
        switch destination {
        case .response:
            if sawResponseEnvelope { throw failure(.duplicateProtocolBlock) }
            if sawToolEnvelope { throw failure(.mixedResponseAndToolCall) }
        case .reasoning:
            if sawResponseEnvelope || sawToolEnvelope {
                throw failure(.mixedResponseAndToolCall)
            }
        case .toolArguments, .toolMarkup:
            if sawResponseEnvelope { throw failure(.mixedResponseAndToolCall) }
        }
    }

    private mutating func flushBody(
        _ text: String,
        to destination: Destination,
        terminatedBy terminator: String?
    ) throws -> [Event] {
        switch destination {
        case .response, .reasoning:
            var events: [Event] = []
            if !text.isEmpty {
                events.append(destination == .reasoning ? .reasoning(text) : .response(text))
            }
            if let terminator { try close(destination, terminatedBy: terminator) }
            return events
        case .toolArguments, .toolMarkup:
            // A structurally incomplete call cannot be dispatched.
            blockBuffer.append(text)
            guard let terminator else { return [] }
            try close(destination, terminatedBy: terminator)
            let body = blockBuffer
            blockBuffer = ""
            let calls = try toolCalls(from: body, to: destination)
            toolCallIndex += calls.count
            return calls
        }
    }

    /// Each recipient is paired with exactly one terminator; a wrong pairing
    /// is a protocol violation, not a shorter message.
    private func expectedTerminator(for destination: Destination) -> String? {
        switch profile {
        case .harmony:
            switch destination {
            case .reasoning: return "<|end|>"
            case .response: return "<|return|>"
            case .toolArguments: return "<|call|>"
            case .toolMarkup: return nil
            }
        case .atem:
            switch destination {
            case .reasoning: return "<|eom|>"
            case .response: return "<|eot|>"
            case .toolArguments, .toolMarkup: return nil
            }
        case .plainChat, .qwen35XML, .gemma4Channels:
            return nil
        }
    }

    private mutating func close(
        _ destination: Destination,
        terminatedBy terminator: String
    ) throws {
        if let expected = expectedTerminator(for: destination), terminator != expected {
            throw failure(.malformedChannel)
        }
        switch destination {
        case .response: sawResponseEnvelope = true
        case .reasoning: break
        case .toolArguments, .toolMarkup: sawToolEnvelope = true
        }
    }

    private func toolCalls(from body: String, to destination: Destination) throws -> [Event] {
        switch destination {
        case .toolArguments(let name):
            return try jsonToolEvents(
                from: "{\"name\":\"\(jsonEscape(name))\",\"arguments\":\(body)}",
                startingAt: toolCallIndex)
        case .toolMarkup:
            return try parseATEMCalls(body, startingAt: toolCallIndex)
        case .response, .reasoning:
            throw failure(.malformedToolCall)
        }
    }

    /// An envelope that never terminated. There is no response-specific
    /// reason code, so a truncated response envelope reports the envelope
    /// itself as malformed.
    private func unfinished(_ destination: Destination) -> CoreAIProtocolFailure {
        switch destination {
        case .response: return .malformedChannel
        case .reasoning: return .unfinishedReasoning
        case .toolArguments, .toolMarkup: return .unfinishedToolCall
        }
    }

    // MARK: - Qwen

    private func parseQwenXMLCall(_ body: String, callIndex: Int) throws -> Event {
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("<function="),
            let nameEnd = trimmed.range(of: ">"),
            let functionClose = trimmed.range(of: "</function>", options: .backwards),
            functionClose.upperBound == trimmed.endIndex
        else { throw failure(.malformedToolCall) }
        let nameStart = trimmed.index(trimmed.startIndex, offsetBy: "<function=".count)
        let name = String(trimmed[nameStart..<nameEnd.lowerBound])
        guard isValidName(name) else { throw failure(.malformedToolCall) }

        var parametersText = String(trimmed[nameEnd.upperBound..<functionClose.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        var arguments: [String: Any] = [:]
        while !parametersText.isEmpty {
            guard parametersText.hasPrefix("<parameter="),
                let keyEnd = parametersText.range(of: ">"),
                let valueEnd = parametersText.range(of: "</parameter>")
            else { throw failure(.malformedToolCall) }
            let keyStart = parametersText.index(
                parametersText.startIndex, offsetBy: "<parameter=".count)
            let key = String(parametersText[keyStart..<keyEnd.lowerBound])
            guard isValidName(key), arguments[key] == nil else {
                throw failure(.duplicateProtocolBlock)
            }
            let raw = String(parametersText[keyEnd.upperBound..<valueEnd.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            arguments[key] = jsonScalarOrString(raw)
            parametersText = String(parametersText[valueEnd.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return try toolEvent(name: name, arguments: arguments, index: callIndex)
    }

    // MARK: - JSON tool profiles

    private func jsonToolEvents(
        from payload: String,
        startingAt start: Int
    ) throws -> [Event] {
        guard let data = payload.trimmingCharacters(in: .whitespacesAndNewlines)
            .data(using: .utf8),
            let json = try? JSONSerialization.jsonObject(with: data)
        else { throw failure(.malformedToolCall) }
        let objects: [[String: Any]]
        if let object = json as? [String: Any] { objects = [object] }
        else if let array = json as? [[String: Any]] { objects = array }
        else { throw failure(.malformedToolCall) }
        guard !objects.isEmpty else { throw failure(.malformedToolCall) }
        return try objects.enumerated().map { offset, object in
            let source = (object["function"] as? [String: Any]) ?? object
            guard let name = source["name"] as? String, isValidName(name) else {
                throw failure(.malformedToolCall)
            }
            let args = source["arguments"] ?? [:]
            guard JSONSerialization.isValidJSONObject(args) else {
                throw failure(.malformedToolCall)
            }
            return try toolEvent(name: name, arguments: args, index: start + offset)
        }
    }

    // MARK: - Gemma 4

    private func parseGemmaCall(_ body: String, callIndex: Int) throws -> Event {
        guard let brace = body.firstIndex(of: "{"), body.hasSuffix("}") else {
            throw failure(.malformedToolCall)
        }
        let name = String(body[..<brace])
        let arguments = try parseGemmaArguments(String(body[brace...]))
        return try toolEvent(name: name, arguments: arguments, index: callIndex)
    }

    private func parseGemmaArguments(_ text: String) throws -> [String: Any] {
        guard text.first == "{", text.last == "}" else { throw failure(.malformedToolCall) }
        let inner = String(text.dropFirst().dropLast())
        if inner.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return [:] }
        var result: [String: Any] = [:]
        for pair in splitTopLevel(inner, separator: ",") {
            let pieces = splitTopLevel(pair, separator: ":", maxSplits: 1)
            guard pieces.count == 2 else { throw failure(.malformedToolCall) }
            let key = pieces[0].trimmingCharacters(in: .whitespacesAndNewlines)
            guard isValidName(key), result[key] == nil else {
                throw failure(.malformedToolCall)
            }
            result[key] = jsonScalarOrString(
                pieces[1].trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return result
    }

    // MARK: - ATEM

    private func parseATEMCalls(_ body: String, startingAt start: Int) throws -> [Event] {
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        let open = "<atem:function_calls>"
        let close = "</atem:function_calls>"
        guard trimmed.hasPrefix(open), trimmed.hasSuffix(close) else {
            throw failure(.malformedToolCall)
        }
        var inner = String(trimmed.dropFirst(open.count).dropLast(close.count))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        var events: [Event] = []
        while !inner.isEmpty {
            let invokePrefix = "<atem:invoke name=\""
            guard inner.hasPrefix(invokePrefix),
                let nameEnd = inner.range(of: "\">") ,
                let invokeClose = inner.range(of: "</atem:invoke>")
            else { throw failure(.malformedToolCall) }
            let nameStart = inner.index(inner.startIndex, offsetBy: invokePrefix.count)
            let name = String(inner[nameStart..<nameEnd.lowerBound])
            var parameters = String(inner[nameEnd.upperBound..<invokeClose.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            var arguments: [String: Any] = [:]
            while !parameters.isEmpty {
                let parameterPrefix = "<atem:parameter name=\""
                guard parameters.hasPrefix(parameterPrefix),
                    let keyEnd = parameters.range(of: "\">"),
                    let valueEnd = parameters.range(of: "</atem:parameter>")
                else { throw failure(.malformedToolCall) }
                let keyStart = parameters.index(
                    parameters.startIndex, offsetBy: parameterPrefix.count)
                let key = String(parameters[keyStart..<keyEnd.lowerBound])
                guard isValidName(key), arguments[key] == nil else {
                    throw failure(.duplicateProtocolBlock)
                }
                let raw = String(parameters[keyEnd.upperBound..<valueEnd.lowerBound])
                arguments[key] = jsonScalarOrString(raw)
                parameters = String(parameters[valueEnd.upperBound...])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            events.append(try toolEvent(name: name, arguments: arguments, index: start + events.count))
            inner = String(inner[invokeClose.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard !events.isEmpty else { throw failure(.malformedToolCall) }
        return events
    }

    // MARK: - Helpers

    private func toolEvent(name: String, arguments: Any, index: Int) throws -> Event {
        guard isValidName(name), JSONSerialization.isValidJSONObject(arguments),
            let data = try? JSONSerialization.data(
                withJSONObject: arguments, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else { throw failure(.malformedToolCall) }
        return .toolCall(id: "coreai-call-\(index + 1)", name: name, argumentsJSON: json)
    }

    private func isValidName(_ name: String) -> Bool {
        guard !name.isEmpty else { return false }
        return name.unicodeScalars.allSatisfy {
            CharacterSet.alphanumerics.contains($0) || $0 == "_" || $0 == "-" || $0 == "."
        }
    }

    private func jsonScalarOrString(_ raw: String) -> Any {
        guard let data = raw.data(using: .utf8),
            let value = try? JSONSerialization.jsonObject(
                with: data, options: [.fragmentsAllowed])
        else { return raw }
        return value
    }

    private func splitTopLevel(
        _ text: String,
        separator: Character,
        maxSplits: Int = .max
    ) -> [String] {
        var result: [String] = []
        var current = ""
        var depth = 0
        var quoted = false
        var escaped = false
        for character in text {
            if quoted {
                current.append(character)
                if escaped { escaped = false }
                else if character == "\\" { escaped = true }
                else if character == "\"" { quoted = false }
                continue
            }
            if character == "\"" { quoted = true; current.append(character) }
            else if character == "{" || character == "[" { depth += 1; current.append(character) }
            else if character == "}" || character == "]" { depth -= 1; current.append(character) }
            else if character == separator, depth == 0, result.count < maxSplits {
                result.append(current)
                current = ""
            } else { current.append(character) }
        }
        result.append(current)
        return result
    }

    private func jsonEscape(_ value: String) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: [value]),
            let encoded = String(data: data, encoding: .utf8)
        else { return "" }
        return String(encoded.dropFirst().dropLast().dropFirst().dropLast())
    }

    private func failure(_ reason: CoreAIProtocolFailure) -> CoreAIProtocolError {
        CoreAIProtocolError(profile: profile, failure: reason)
    }
}
