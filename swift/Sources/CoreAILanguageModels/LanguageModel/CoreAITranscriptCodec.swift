// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import FoundationModels
import Tokenizers

/// Native-Jinja transcript encoder shared by the Core AI text and vision
/// adapters. It has no plain-text or model-family fallback path.
package struct CoreAITranscriptCodec {
    package struct EncodedTranscript {
        package let tokens: [Int]
        package let images: [Transcript.ImageAttachment]
    }

    package struct ReasoningConfiguration {
        package let enabled: Bool
        package let additionalContext: [String: any Sendable]?
    }

    private let profile: CoreAILanguageProtocolProfile

    package init(profile: CoreAILanguageProtocolProfile) {
        self.profile = profile
    }

    /// Validates the selected profile's required template behavior. This must
    /// run after tokenizer loading and before any inference engine is loaded.
    package func validate(tokenizer: any Tokenizer) throws {
        guard tokenizer.hasChatTemplate else { throw failure(.missingChatTemplate) }

        let fixtureTools = [Self.fixtureTool]
        let fixtureMessages = fixtureMessages()
        let additionalContext = validationContext()
        let tokens: [Int]
        do {
            tokens = try tokenizer.applyChatTemplate(
                messages: fixtureMessages,
                tools: profile.supportsToolCalling ? fixtureTools : nil,
                additionalContext: additionalContext)
        } catch {
            throw failure(.incompatibleChatTemplate)
        }
        guard !tokens.isEmpty else { throw failure(.incompatibleChatTemplate) }

        let rendered = tokenizer.decode(tokens: tokens)
        try assertEvidence(in: rendered)
    }

    package func encode(
        entries: [Transcript.Entry],
        tools: [Transcript.ToolDefinition],
        reasoning: ReasoningConfiguration,
        using tokenizer: any Tokenizer
    ) throws -> EncodedTranscript {
        guard tokenizer.hasChatTemplate else { throw failure(.missingChatTemplate) }
        var messages: [Message] = []
        var images: [Transcript.ImageAttachment] = []
        var pendingReasoning: String?

        for entry in entries {
            switch entry {
            case .instructions(let instructions):
                let text = try textOnly(instructions.segments)
                if !text.isEmpty { messages.append(["role": "system", "content": text]) }

            case .prompt(let prompt):
                let content = try promptContent(prompt.segments, images: &images)
                messages.append(["role": "user", "content": content])

            case .reasoning(let reasoningEntry):
                guard profile.supportsReasoning, pendingReasoning == nil else {
                    throw failure(.unsupportedTranscriptContent)
                }
                let text = try textOnly(reasoningEntry.segments)
                guard !text.isEmpty else { throw failure(.unsupportedTranscriptContent) }
                pendingReasoning = text

            case .response(let response):
                let text = try textOnly(response.segments)
                var message: Message = ["role": "assistant", "content": text]
                attach(reasoning: pendingReasoning, to: &message)
                pendingReasoning = nil
                messages.append(message)

            case .toolCalls(let toolCalls):
                guard profile.supportsToolCalling, !toolCalls.isEmpty else {
                    throw failure(.unsupportedTranscriptContent)
                }
                let calls = try toolCalls.map(toolCallMessage)
                var message: Message = [
                    "role": "assistant",
                    "content": "" as any Sendable,
                    "tool_calls": calls as any Sendable,
                ]
                attach(reasoning: pendingReasoning, to: &message)
                pendingReasoning = nil
                messages.append(message)

            case .toolOutput(let output):
                guard profile.supportsToolCalling else {
                    throw failure(.unsupportedTranscriptContent)
                }
                messages.append([
                    "role": toolOutputRole,
                    "tool_call_id": normalizedToolCallID(output.id),
                    "name": output.toolName,
                    "content": try textOnly(output.segments),
                ])

            @unknown default:
                throw failure(.unsupportedTranscriptContent)
            }
        }

        guard pendingReasoning == nil, !messages.isEmpty else {
            throw failure(.unsupportedTranscriptContent)
        }
        let toolSpecs = try tools.map(makeToolSpec)
        do {
            let tokens = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: toolSpecs.isEmpty ? nil : toolSpecs,
                additionalContext: mergedContext(
                    reasoning.additionalContext,
                    preservingReasoning: entries.containsReasoning))
            guard !tokens.isEmpty else { throw failure(.incompatibleChatTemplate) }
            return EncodedTranscript(tokens: tokens, images: images)
        } catch let error as CoreAIProtocolError {
            throw error
        } catch {
            throw failure(.incompatibleChatTemplate)
        }
    }

    package func reasoningConfiguration(
        for level: ContextOptions.ReasoningLevel?
    ) throws -> ReasoningConfiguration {
        guard let level else {
            return ReasoningConfiguration(
                enabled: profile.defaultReasoningEnabled,
                additionalContext: defaultReasoningContext)
        }
        guard profile.supportsReasoning else { throw failure(.unsupportedReasoningPolicy) }
        switch level {
        case .light:
            return ReasoningConfiguration(enabled: true, additionalContext: enabledContext("low"))
        case .moderate:
            return ReasoningConfiguration(enabled: true, additionalContext: enabledContext("medium"))
        case .deep:
            return ReasoningConfiguration(enabled: true, additionalContext: enabledContext("high"))
        case .custom(let value):
            let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            guard normalized == "no_think", profile.supportsDisablingReasoning else {
                throw failure(.unsupportedReasoningPolicy)
            }
            return ReasoningConfiguration(
                enabled: false,
                additionalContext: ["enable_thinking": false])
        @unknown default:
            throw failure(.unsupportedReasoningPolicy)
        }
    }

    package func expandImagePlaceholder(
        in tokens: [Int],
        imageTokenID: Int32,
        imageTokenCount: Int
    ) throws -> [Int32] {
        let positions = tokens.indices.filter { tokens[$0] == Int(imageTokenID) }
        guard positions.count == 1 else {
            throw failure(positions.isEmpty ? .missingImagePlaceholder : .duplicateImagePlaceholder)
        }
        var result: [Int32] = []
        result.reserveCapacity(tokens.count + imageTokenCount - 1)
        for (index, token) in tokens.enumerated() {
            if index == positions[0] {
                result.append(contentsOf: repeatElement(imageTokenID, count: imageTokenCount))
            } else {
                result.append(Int32(token))
            }
        }
        return result
    }

    /// Confirms this profile's template renders exactly one image
    /// placeholder, so a VLM bundle is only paired with a profile that has
    /// an image convention. Called during validation, before engine load.
    ///
    /// The fixture mirrors `CoreAIVisionLanguageModel.buildPromptTokens`,
    /// which resolves the image token via
    /// `tokenizer.convertIdToToken(imageTokenId)` (falling back to
    /// `"<|image_pad|>"`), composes the prompt as
    /// `"\(imageToken)\n\(userText)"`, and applies the chat template to a
    /// single user message holding that string. Validating anything else —
    /// a text-only fixture, a different composition — would prove nothing
    /// about what production actually renders.
    package func validateVisionPairing(
        tokenizer: any Tokenizer,
        imageTokenID: Int32
    ) throws {
        let imageToken = tokenizer.convertIdToToken(Int(imageTokenID)) ?? "<|image_pad|>"
        let messages: [Message] = [
            ["role": "user", "content": "\(imageToken)\nsynthetic-user"]
        ]
        let tokens: [Int]
        do {
            tokens = try tokenizer.applyChatTemplate(messages: messages)
        } catch {
            throw failure(.incompatibleChatTemplate)
        }
        let count = tokens.filter { $0 == Int(imageTokenID) }.count
        guard count == 1 else {
            throw failure(count == 0 ? .missingImagePlaceholder : .duplicateImagePlaceholder)
        }
    }

    // MARK: - Transcript messages

    private func textOnly(_ segments: [Transcript.Segment]) throws -> String {
        var result = ""
        for segment in segments {
            guard case .text(let text) = segment else {
                throw failure(.unsupportedTranscriptContent)
            }
            result += text.content
        }
        return result
    }

    private func promptContent(
        _ segments: [Transcript.Segment],
        images: inout [Transcript.ImageAttachment]
    ) throws -> any Sendable {
        var parts: [[String: any Sendable]] = []
        var textOnly = ""
        var hasAttachment = false
        for segment in segments {
            switch segment {
            case .text(let text):
                textOnly += text.content
                parts.append(["type": "text", "text": text.content])
            case .attachment(let attachment):
                guard case .image(let image) = attachment.content else {
                    throw failure(.unsupportedTranscriptContent)
                }
                hasAttachment = true
                images.append(image)
                parts.append(["type": "image"])
            @unknown default:
                throw failure(.unsupportedTranscriptContent)
            }
        }
        return hasAttachment ? parts : textOnly
    }

    private func attach(reasoning: String?, to message: inout Message) {
        guard let reasoning else { return }
        message[reasoningMessageKey] = reasoning
    }

    /// The single source of truth for which transcript key carries reasoning
    /// content for the selected profile. Used by both live encoding and
    /// template-validation fixtures so the two can never diverge.
    private var reasoningMessageKey: String {
        switch profile {
        case .harmony: "thinking"
        case .plainChat, .qwen35XML, .gemma4Channels, .atem: "reasoning_content"
        }
    }

    private func toolCallMessage(
        _ call: Transcript.ToolCall
    ) throws -> [String: any Sendable] {
        guard let arguments = Self.sendableJSONObject(from: call.arguments.jsonString) as? [String: any Sendable]
        else { throw failure(.invalidPriorToolCall) }
        let function: [String: any Sendable] = [
            "name": call.toolName,
            "arguments": arguments,
        ]
        return [
            "id": normalizedToolCallID(call.id),
            "type": "function",
            "function": function,
        ]
    }

    private var toolOutputRole: String {
        "tool"
    }

    private func normalizedToolCallID(_ id: String) -> String {
        id
    }

    private func makeToolSpec(_ definition: Transcript.ToolDefinition) throws -> ToolSpec {
        guard let schemaData = try? JSONEncoder().encode(definition.parameters),
            let object = try? JSONSerialization.jsonObject(with: schemaData),
            let parameters = Self.sendableJSONValue(object) as? [String: any Sendable]
        else { throw failure(.invalidToolDefinition) }
        let function: [String: any Sendable] = [
            "name": definition.name,
            "description": definition.description,
            "parameters": parameters,
        ]
        return ["type": "function", "function": function]
    }

    // MARK: - Template validation

    private func fixtureMessages() -> [Message] {
        var assistant: Message = [
            "role": "assistant",
            "content": "synthetic-response",
        ]
        if profile.supportsReasoning {
            assistant[reasoningMessageKey] = "synthetic-reasoning"
        }
        if profile.supportsToolCalling {
            assistant["content"] = ""
            assistant["tool_calls"] = [[
                "id": "synthetic-id",
                "type": "function",
                "function": [
                    "name": "synthetic.tool",
                    "arguments": ["value": 1] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable]] as any Sendable
        }
        var messages: [Message] = [
            ["role": "system", "content": "synthetic-system"],
            ["role": "user", "content": "synthetic-user"],
            assistant,
        ]
        if profile.supportsToolCalling {
            messages.append([
                "role": toolOutputRole,
                "tool_call_id": "synthetic-id",
                "name": "synthetic.tool",
                "content": "synthetic-result",
            ])
            messages.append(["role": "user", "content": "synthetic-follow-up"])
        }
        return messages
    }

    private static let fixtureTool: ToolSpec = [
        "type": "function",
        "function": [
            "name": "synthetic.tool",
            "description": "Synthetic validation tool",
            "parameters": [
                "type": "object",
                "properties": ["value": ["type": "integer"] as [String: any Sendable]],
                "required": ["value"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    /// Every marker owned by a structural profile. Selecting `plainChat` for
    /// an artifact whose template emits any of these is rejected at load
    /// rather than at first generation.
    package static let reservedMarkers: [String] = [
        "<think>", "</think>",
        "<tool_call>", "</tool_call>",
        "<function=", "<parameter=",
        "<|channel|>", "<|channel>", "<channel|>",
        "<|tool_call>", "<tool_call|>",
        "<|start|>", "<|message|>", "<|end|>", "<|return|>", "<|call|>",
        "<|eom|>", "<|eot|>",
        "<atem:function_calls>",
    ]

    /// Strings proving the template rendered reasoning content. Empty only
    /// for profiles that do not advertise reasoning.
    package var reasoningEvidence: [String] {
        switch profile {
        case .plainChat: []
        case .qwen35XML: ["<think>", "synthetic-reasoning"]
        case .harmony: ["<|channel|>analysis", "synthetic-reasoning"]
        case .gemma4Channels: ["<|channel>thought", "synthetic-reasoning"]
        case .atem: ["to=self", "synthetic-reasoning"]
        }
    }

    /// Strings proving the template rendered tool definitions, a tool call,
    /// and a prior tool result. Empty only for profiles without tools.
    package var toolEvidence: [String] {
        switch profile {
        case .plainChat: []
        case .qwen35XML: ["<function=synthetic.tool>", "<parameter=value>", "synthetic-result"]
        case .harmony: ["to=functions.synthetic.tool", "synthetic-result"]
        case .gemma4Channels: ["<|tool_call>call:synthetic.tool", "synthetic-result"]
        case .atem: ["<atem:function_calls>", "synthetic-result"]
        }
    }

    /// Strings proving the template rendered ordinary roles and content.
    private var contentEvidence: [String] {
        ["synthetic-system", "synthetic-user", "synthetic-response"]
    }

    package var templateEvidence: [String] {
        switch profile {
        case .plainChat: contentEvidence
        case .qwen35XML, .harmony, .gemma4Channels, .atem:
            reasoningEvidence + toolEvidence
        }
    }

    /// Asserts the rendered fixture carries every required string, and for
    /// `plainChat` that it carries no structural marker at all.
    package func assertEvidence(in rendered: String) throws {
        guard templateEvidence.allSatisfy(rendered.contains) else {
            throw failure(.incompatibleChatTemplate)
        }
        if profile == .plainChat,
            Self.reservedMarkers.contains(where: rendered.contains) {
            throw failure(.incompatibleChatTemplate)
        }
    }

    private func validationContext() -> [String: any Sendable]? {
        var context = enabledContext("high") ?? [:]
        if profile == .qwen35XML || profile == .gemma4Channels {
            context["preserve_thinking"] = true
        }
        return context.isEmpty ? nil : context
    }

    private var defaultReasoningContext: [String: any Sendable]? {
        switch profile {
        case .qwen35XML:
            ["enable_thinking": true]
        case .harmony:
            ["reasoning_effort": "medium"]
        case .gemma4Channels:
            ["enable_thinking": false]
        case .atem:
            ["reasoning_strength": "high"]
        case .plainChat:
            nil
        }
    }

    private func enabledContext(_ effort: String) -> [String: any Sendable]? {
        switch profile {
        case .qwen35XML, .gemma4Channels:
            ["enable_thinking": true]
        case .harmony:
            ["reasoning_effort": effort]
        case .atem:
            ["reasoning_strength": effort]
        case .plainChat:
            nil
        }
    }

    private func mergedContext(
        _ context: [String: any Sendable]?,
        preservingReasoning: Bool
    ) -> [String: any Sendable]? {
        var result = context ?? [:]
        if preservingReasoning,
            profile == .qwen35XML || profile == .gemma4Channels
        {
            result["preserve_thinking"] = true
        }
        return result.isEmpty ? nil : result
    }

    private static func sendableJSONObject(from string: String) -> (any Sendable)? {
        guard let data = string.data(using: .utf8),
            let value = try? JSONSerialization.jsonObject(with: data)
        else { return nil }
        return sendableJSONValue(value)
    }

    private static func sendableJSONValue(_ value: Any) -> any Sendable {
        switch value {
        case let value as String: value
        case let value as NSNumber where CFGetTypeID(value) == CFBooleanGetTypeID(): value.boolValue
        case let value as NSNumber:
            value.doubleValue == value.doubleValue.rounded() ? value.intValue : value.doubleValue
        case let value as [Any]: value.map(sendableJSONValue)
        case let value as [String: Any]: value.mapValues(sendableJSONValue)
        default: NSNull()
        }
    }

    private func failure(_ reason: CoreAIProtocolFailure) -> CoreAIProtocolError {
        CoreAIProtocolError(profile: profile, failure: reason)
    }
}

private extension Array where Element == Transcript.Entry {
    var containsReasoning: Bool {
        contains {
            if case .reasoning = $0 { return true }
            return false
        }
    }
}
