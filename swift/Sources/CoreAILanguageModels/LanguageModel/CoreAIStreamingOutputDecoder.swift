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
/// incomplete call cannot be dispatched. Envelope profiles (`harmony`, `atem`)
/// are still decoded whole at native termination.
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
        /// A tool body that opened without its wrapping block marker. Its text
        /// is swallowed rather than passed off as a response, and the stream
        /// fails at termination.
        case strayToolBlock
    }

    private let profile: CoreAILanguageProtocolProfile
    private let reasoningEnabled: Bool
    private var scanner = MarkerScanner()
    private var state: State = .text
    private var blockBuffer = ""
    private var sawReasoning = false
    private var trimLeadingReasoningNewline = false
    private var toolCallIndex = 0

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

    /// Inner tool marker that is only legal inside `toolOpen`. Seeing it at top
    /// level means a malformed call, which must be withheld rather than
    /// streamed as response text.
    private var strayToolOpen: String? {
        switch profile {
        case .qwen35XML: "<function="
        case .plainChat, .gemma4Channels, .harmony, .atem: nil
        }
    }

    /// Markers that change state when matched in the current state.
    private var pendingMarkers: [String] {
        switch state {
        case .text: [reasoningOpen, reasoningClose, toolOpen, strayToolOpen].compactMap { $0 }
        case .reasoning: [reasoningClose, reasoningOpen].compactMap { $0 }
        case .toolBlock: [toolClose].compactMap { $0 }
        case .strayToolBlock: []
        }
    }

    /// Markers whose partial suffix must be withheld from the caller. Equal to
    /// `pendingMarkers` except for `plainChat`, which has no structural markers
    /// of its own yet must never let a reserved marker through split across two
    /// deltas.
    private var holdBackMarkers: [String] {
        let pending = pendingMarkers
        guard pending.isEmpty else { return pending }
        if profile == .plainChat, state == .text { return CoreAITranscriptCodec.reservedMarkers }
        return []
    }

    // MARK: - Inline drain

    private mutating func drainInline(isFinal: Bool) throws -> [Event] {
        var events: [Event] = []
        while true {
            if let match = scanner.firstMatch(of: pendingMarkers) {
                let before = scanner.takeUpTo(match.range)
                events += try emit(before)
                try transition(on: match.marker, into: &events)
                continue
            }
            let safe = scanner.takeSafe(waitingFor: holdBackMarkers, isFinal: isFinal)
            events += try emit(safe)
            if isFinal { try assertClosed() }
            return events
        }
    }

    private mutating func emit(_ text: String) throws -> [Event] {
        guard !text.isEmpty else { return [] }
        switch state {
        case .text:
            try rejectReservedMarkers(in: text)
            return [.response(text)]
        case .reasoning:
            var body = text
            if trimLeadingReasoningNewline {
                trimLeadingReasoningNewline = false
                if body.hasPrefix("\n") { body.removeFirst() }
            }
            return body.isEmpty ? [] : [.reasoning(body)]
        case .toolBlock, .strayToolBlock:
            blockBuffer.append(text)
            return []
        }
    }

    private mutating func transition(on marker: String, into events: inout [Event]) throws {
        if marker == reasoningOpen {
            guard reasoningEnabled else { throw failure(.malformedReasoning) }
            guard state != .reasoning else { throw failure(.nestedProtocolBlock) }
            guard !sawReasoning else { throw failure(.duplicateProtocolBlock) }
            guard state == .text else { throw failure(.nestedProtocolBlock) }
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
        } else if marker == strayToolOpen {
            guard state == .text else { throw failure(.nestedProtocolBlock) }
            blockBuffer = ""
            state = .strayToolBlock
        }
    }

    private func assertClosed() throws {
        switch state {
        case .text: return
        case .reasoning: throw failure(.unfinishedReasoning)
        case .toolBlock: throw failure(.unfinishedToolCall)
        case .strayToolBlock: throw failure(.malformedToolCall)
        }
    }

    /// Guards against a structural marker appearing in what should be plain
    /// response text — the `plainChat`-on-a-reasoning-model case.
    private func rejectReservedMarkers(in text: String) throws {
        guard profile == .plainChat else { return }
        guard !CoreAITranscriptCodec.reservedMarkers.contains(where: text.contains) else {
            throw failure(.malformedChannel)
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

    // MARK: - Envelope drain

    /// TEMPORARY (Task 5): a shim that preserves the pre-existing whole-output
    /// behavior of the envelope profiles. Task 6 replaces it with real
    /// envelope streaming.
    private mutating func drainEnvelope(isFinal: Bool) throws -> [Event] {
        guard isFinal else { return [] }
        let whole = scanner.takeAll()
        switch profile {
        case .harmony: return try parseHarmony(whole)
        case .atem: return try parseATEM(whole)
        case .plainChat, .qwen35XML, .gemma4Channels: return []
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

    // MARK: - Harmony

    private func parseHarmony(_ original: String) throws -> [Event] {
        var remainder = original.trimmingCharacters(in: .whitespacesAndNewlines)
        var events: [Event] = []
        var hasFinal = false
        var hasTool = false
        while !remainder.isEmpty {
            guard remainder.hasPrefix("<|start|>assistant"),
                let messageMarker = remainder.range(of: "<|message|>")
            else { throw failure(.malformedChannel) }
            let headerStart = remainder.index(
                remainder.startIndex, offsetBy: "<|start|>assistant".count)
            let header = String(remainder[headerStart..<messageMarker.lowerBound])
            let bodyStart = messageMarker.upperBound
            guard let termination = firstTermination(
                in: remainder, after: bodyStart,
                markers: ["<|end|>", "<|return|>", "<|call|>"])
            else { throw failure(.malformedChannel) }
            let body = String(remainder[bodyStart..<termination.range.lowerBound])

            if header == "<|channel|>analysis" {
                guard !hasFinal, !hasTool, termination.marker == "<|end|>" else {
                    throw failure(.mixedResponseAndToolCall)
                }
                if !body.isEmpty { events.append(.reasoning(body)) }
            } else if header == "<|channel|>final" {
                guard !hasFinal, !hasTool, termination.marker == "<|return|>" else {
                    throw failure(.duplicateProtocolBlock)
                }
                hasFinal = true
                if !body.isEmpty { events.append(.response(body)) }
            } else if header.hasPrefix(" to=functions."),
                let channel = header.range(of: "<|channel|>commentary")
            {
                guard !hasFinal, termination.marker == "<|call|>" else {
                    throw failure(.mixedResponseAndToolCall)
                }
                hasTool = true
                let nameStart = header.index(header.startIndex, offsetBy: " to=functions.".count)
                let name = String(header[nameStart..<channel.lowerBound])
                let calls = try jsonToolEvents(
                    from: "{\"name\":\"\(jsonEscape(name))\",\"arguments\":\(body)}",
                    startingAt: events.toolCallCount)
                events.append(contentsOf: calls)
            } else {
                throw failure(.malformedChannel)
            }
            remainder = String(remainder[termination.range.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return events
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

    private func parseATEM(_ original: String) throws -> [Event] {
        var remainder = original.trimmingCharacters(in: .whitespacesAndNewlines)
        var events: [Event] = []
        var hasFinal = false
        var hasTool = false
        while !remainder.isEmpty {
            let prefix = "<|start|>assistant to="
            guard remainder.hasPrefix(prefix),
                let messageMarker = remainder.range(of: "<|message|>")
            else { throw failure(.malformedChannel) }
            let recipientStart = remainder.index(remainder.startIndex, offsetBy: prefix.count)
            let recipient = String(remainder[recipientStart..<messageMarker.lowerBound])
            guard let termination = firstTermination(
                in: remainder, after: messageMarker.upperBound,
                markers: ["<|eom|>", "<|eot|>"])
            else { throw failure(.malformedChannel) }
            let body = String(remainder[messageMarker.upperBound..<termination.range.lowerBound])
            switch recipient {
            case "self":
                guard !hasFinal, !hasTool, termination.marker == "<|eom|>" else {
                    throw failure(.mixedResponseAndToolCall)
                }
                if !body.isEmpty { events.append(.reasoning(body)) }
            case "user":
                guard !hasFinal, !hasTool, termination.marker == "<|eot|>" else {
                    throw failure(.duplicateProtocolBlock)
                }
                hasFinal = true
                if !body.isEmpty { events.append(.response(body)) }
            default:
                guard !hasFinal else { throw failure(.mixedResponseAndToolCall) }
                hasTool = true
                events.append(contentsOf: try parseATEMCalls(body, startingAt: events.toolCallCount))
            }
            remainder = String(remainder[termination.range.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return events
    }

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

    private func firstTermination(
        in text: String,
        after start: String.Index,
        markers: [String]
    ) -> (marker: String, range: Range<String.Index>)? {
        markers.compactMap { marker in
            text.range(of: marker, range: start..<text.endIndex).map { (marker, $0) }
        }.min { $0.1.lowerBound < $1.1.lowerBound }
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

private extension Array where Element == CoreAIStreamingOutputDecoder.Event {
    var toolCallCount: Int {
        reduce(into: 0) { count, event in
            if case .toolCall = event { count += 1 }
        }
    }
}
