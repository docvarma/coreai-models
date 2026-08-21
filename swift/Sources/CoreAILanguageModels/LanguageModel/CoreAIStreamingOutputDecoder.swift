// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Strict decoder shared by the Core AI text and vision adapters.
///
/// Generated deltas are retained until native termination so a later malformed,
/// nested, duplicate, or mixed protocol block can fail before content is
/// committed to a FoundationModels transcript. This also makes every reserved
/// marker safe when split at any streaming boundary.
package struct CoreAIStreamingOutputDecoder {
    package enum Event: Equatable, Sendable {
        case response(String)
        case reasoning(String)
        case toolCall(id: String, name: String, argumentsJSON: String)
    }

    private let profile: CoreAILanguageProtocolProfile
    private let reasoningEnabled: Bool
    private var buffer = ""

    package init(
        profile: CoreAILanguageProtocolProfile,
        reasoningEnabled: Bool
    ) {
        self.profile = profile
        self.reasoningEnabled = reasoningEnabled
    }

    package mutating func consume(_ delta: String) throws -> [Event] {
        buffer.append(delta)
        return []
    }

    package mutating func finish() throws -> [Event] {
        defer { buffer.removeAll(keepingCapacity: false) }
        switch profile {
        case .plainChat:
            return try parsePlain(buffer)
        case .qwen35XML:
            return try parseQwenXML(buffer)
        case .harmony:
            return try parseHarmony(buffer)
        case .gemma4Channels:
            return try parseGemma(buffer)
        case .atem:
            return try parseATEM(buffer)
        }
    }

    // MARK: - Plain profiles

    private func parsePlain(_ text: String) throws -> [Event] {
        guard !CoreAITranscriptCodec.reservedMarkers.contains(where: text.contains) else {
            throw failure(.malformedChannel)
        }
        return text.isEmpty ? [] : [.response(text)]
    }

    // MARK: - Qwen

    private func parseQwenXML(_ original: String) throws -> [Event] {
        var text = original
        var events: [Event] = []
        if reasoningEnabled {
            if text.hasPrefix("<think>\n") { text.removeFirst("<think>\n".count) }
            else if text.hasPrefix("<think>") { text.removeFirst("<think>".count) }
            guard let close = text.range(of: "</think>") else {
                throw failure(.unfinishedReasoning)
            }
            let reasoning = String(text[..<close.lowerBound])
            guard !reasoning.contains("<think>") else { throw failure(.nestedProtocolBlock) }
            if !reasoning.isEmpty { events.append(.reasoning(reasoning)) }
            text = String(text[close.upperBound...])
            guard !text.contains("<think>"), !text.contains("</think>") else {
                throw failure(.duplicateProtocolBlock)
            }
        } else if text.contains("<think>") || text.contains("</think>") {
            throw failure(.malformedReasoning)
        }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.contains("<tool_call>") else {
            return events + (text.isEmpty ? [] : [.response(text)])
        }
        guard text.replacingOccurrences(of: "<tool_call>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .hasPrefix("<function=")
        else { throw failure(.mixedResponseAndToolCall) }

        var remainder = trimmed
        var callIndex = 0
        while !remainder.isEmpty {
            guard remainder.hasPrefix("<tool_call>"),
                let close = remainder.range(of: "</tool_call>")
            else { throw failure(.unfinishedToolCall) }
            let bodyStart = remainder.index(remainder.startIndex, offsetBy: "<tool_call>".count)
            let body = String(remainder[bodyStart..<close.lowerBound])
            let call = try parseQwenXMLCall(body, callIndex: callIndex)
            events.append(call)
            callIndex += 1
            remainder = String(remainder[close.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return events
    }

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

    private func parseGemma(_ original: String) throws -> [Event] {
        var text = original
        var events: [Event] = []
        if reasoningEnabled {
            let open = "<|channel>thought\n"
            guard text.hasPrefix(open), let close = text.range(of: "<channel|>") else {
                throw failure(.unfinishedReasoning)
            }
            let start = text.index(text.startIndex, offsetBy: open.count)
            let reasoning = String(text[start..<close.lowerBound])
            if !reasoning.isEmpty { events.append(.reasoning(reasoning)) }
            text = String(text[close.upperBound...])
        } else if text.contains("<|channel>thought") || text.contains("<channel|>") {
            throw failure(.malformedReasoning)
        }
        guard !text.contains("<|channel>thought"), !text.contains("<channel|>") else {
            throw failure(.duplicateProtocolBlock)
        }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.contains("<|tool_call>") else {
            return events + (text.isEmpty ? [] : [.response(text)])
        }
        guard trimmed.hasPrefix("<|tool_call>") else {
            throw failure(.mixedResponseAndToolCall)
        }
        var remainder = trimmed
        var callIndex = 0
        while !remainder.isEmpty {
            let open = "<|tool_call>call:"
            guard remainder.hasPrefix(open),
                let close = remainder.range(of: "<tool_call|>")
            else { throw failure(.unfinishedToolCall) }
            let start = remainder.index(remainder.startIndex, offsetBy: open.count)
            let call = String(remainder[start..<close.lowerBound])
            guard let brace = call.firstIndex(of: "{"), call.hasSuffix("}") else {
                throw failure(.malformedToolCall)
            }
            let name = String(call[..<brace])
            let argumentText = String(call[brace...])
            let arguments = try parseGemmaArguments(argumentText)
            events.append(try toolEvent(name: name, arguments: arguments, index: callIndex))
            callIndex += 1
            remainder = String(remainder[close.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return events
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
