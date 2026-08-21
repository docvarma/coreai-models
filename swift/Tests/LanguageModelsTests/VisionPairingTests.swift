// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing
import Tokenizers

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

@Suite("Vision pairing")
struct VisionPairingTests {
    @Test("Exactly one placeholder is required")
    func exactlyOnePlaceholder() throws {
        let codec = CoreAITranscriptCodec(profile: .qwen35XML)
        let expanded = try codec.expandImagePlaceholder(
            in: [1, 99, 2], imageTokenID: 99, imageTokenCount: 3)
        #expect(expanded == [1, 99, 99, 99, 2])
    }

    @Test("A missing placeholder is rejected")
    func missingPlaceholderRejected() {
        let codec = CoreAITranscriptCodec(profile: .qwen35XML)
        #expect(throws: CoreAIProtocolError.self) {
            try codec.expandImagePlaceholder(in: [1, 2], imageTokenID: 99, imageTokenCount: 3)
        }
    }

    @Test("A duplicate placeholder is rejected")
    func duplicatePlaceholderRejected() {
        let codec = CoreAITranscriptCodec(profile: .qwen35XML)
        #expect(throws: CoreAIProtocolError.self) {
            try codec.expandImagePlaceholder(
                in: [99, 1, 99], imageTokenID: 99, imageTokenCount: 3)
        }
    }

    @Test("Failure codes distinguish missing from duplicate")
    func failureCodesAreDistinct() {
        let codec = CoreAITranscriptCodec(profile: .qwen35XML)
        #expect(throws: CoreAIProtocolError(profile: .qwen35XML, failure: .missingImagePlaceholder)) {
            try codec.expandImagePlaceholder(in: [1], imageTokenID: 99, imageTokenCount: 2)
        }
        #expect(throws: CoreAIProtocolError(profile: .qwen35XML, failure: .duplicateImagePlaceholder)) {
            try codec.expandImagePlaceholder(in: [99, 99], imageTokenID: 99, imageTokenCount: 2)
        }
    }

    // MARK: - validateVisionPairing

    // These exercise `validateVisionPairing`, the gate that proves a profile's
    // template actually has an image convention before a VLM bundle is
    // allowed to pair with it. Each test drives the fixture through a stub
    // tokenizer whose `render` closure sees exactly what
    // `validateVisionPairing` composed and applied via
    // `tokenizer.applyChatTemplate(messages:)` — the same call shape
    // `CoreAIVisionLanguageModel.buildPromptTokens` uses in production.

    @Test("The vision fixture mirrors production's image-token + newline + text shape")
    func exactlyOnePairingSucceeds() throws {
        let codec = CoreAITranscriptCodec(profile: .qwen35XML)
        let imageTokenID: Int32 = 4242
        // `buildPromptTokens` in CoreAIVisionLanguageModel composes
        // "\(imageToken)\n\(userText)" before applying the chat template.
        // Only that exact shape causes this stub to emit the placeholder
        // token, so a passing test here proves the fixture matches
        // production's construction rather than merely rendering *something*.
        let tokenizer = FakeVisionTokenizer(imageTokenText: "<img>") { content in
            content == "<img>\nsynthetic-user" ? [1, Int(imageTokenID), 2] : [1, 2]
        }
        try codec.validateVisionPairing(tokenizer: tokenizer, imageTokenID: imageTokenID)
    }

    @Test("A profile with no image convention is rejected as missing")
    func noImageConventionIsRejected() {
        let codec = CoreAITranscriptCodec(profile: .qwen35XML)
        let imageTokenID: Int32 = 4242
        // No token in the vocabulary maps back to `imageTokenID`, so the
        // fixture must fall back to a literal placeholder string — and this
        // template has no convention for rendering it, so it never appears
        // in the output.
        let tokenizer = FakeVisionTokenizer(imageTokenText: nil) { _ in [1, 2, 3] }
        #expect(throws: CoreAIProtocolError(profile: .qwen35XML, failure: .missingImagePlaceholder)) {
            try codec.validateVisionPairing(tokenizer: tokenizer, imageTokenID: imageTokenID)
        }
    }

    @Test("A template that expands the placeholder twice is rejected as duplicate")
    func duplicateExpansionIsRejected() {
        let codec = CoreAITranscriptCodec(profile: .qwen35XML)
        let imageTokenID: Int32 = 4242
        let tokenizer = FakeVisionTokenizer(imageTokenText: "<img>") { _ in
            [1, Int(imageTokenID), 2, Int(imageTokenID), 3]
        }
        #expect(throws: CoreAIProtocolError(profile: .qwen35XML, failure: .duplicateImagePlaceholder)) {
            try codec.validateVisionPairing(tokenizer: tokenizer, imageTokenID: imageTokenID)
        }
    }

    @Test("A chat template that fails to apply is reported as incompatible")
    func templateApplicationFailureIsReported() {
        let codec = CoreAITranscriptCodec(profile: .qwen35XML)
        struct StubTemplateFailure: Error {}
        let tokenizer = FakeVisionTokenizer(imageTokenText: "<img>") { _ in
            throw StubTemplateFailure()
        }
        #expect(throws: CoreAIProtocolError(profile: .qwen35XML, failure: .incompatibleChatTemplate)) {
            try codec.validateVisionPairing(tokenizer: tokenizer, imageTokenID: 4242)
        }
    }
}

/// A minimal `Tokenizer` stub for vision-pairing tests. `render` receives the
/// exact joined message content `validateVisionPairing` composed, and
/// controls the returned token stream — letting each test assert both what
/// was rendered and how many times the image placeholder appears in the
/// result, independent of any real chat-template engine.
private struct FakeVisionTokenizer: Tokenizer, Sendable {
    let imageTokenText: String?
    let render: @Sendable (String) throws -> [Int]

    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { nil }
    var eosTokenId: Int? { nil }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }

    func tokenize(text: String) -> [String] { [text] }
    func encode(text: String) -> [Int] { [] }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func callAsFunction(_ text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokens: [Int]) -> String { "" }
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertTokensToIds(_ tokens: [String]) -> [Int?] { tokens.map { _ in nil } }
    func convertIdToToken(_ id: Int) -> String? { imageTokenText }
    func convertIdsToTokens(_ ids: [Int]) -> [String?] { ids.map { _ in imageTokenText } }

    func applyChatTemplate(messages: [Message]) throws -> [Int] {
        let content = messages.compactMap { $0["content"] as? String }.joined(separator: "\n")
        return try render(content)
    }

    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(
        messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(
        messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool,
        truncation: Bool, maxLength: Int?, tools: [ToolSpec]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(
        messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool,
        truncation: Bool, maxLength: Int?, tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }
}

#endif
