// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing
import Tokenizers

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

/// The VLM path used to build its stop set from `eosTokenId` plus a hardcoded
/// `<|im_end|>`. That literal is a model-family guess, and an artifact whose
/// family ends a turn some other way ran straight past its own end of turn:
/// the raw end-of-turn marker reached the caller as response text, and the
/// model's fabricated next turn followed it into the same pane. These drive
/// `CoreAIVLMExecutor`'s real generation loop with a synthetic token sequence,
/// so nothing here needs a model.
@Suite("VLM end of turn")
struct VLMEndOfTurnTests {
    /// Synthetic vocabulary. `20` is the artifact's declared end-of-turn token
    /// and stands in for a family-specific spelling; `77` is the spelling the
    /// deleted guess hardcoded, present in the vocabulary but *not* declared
    /// by this artifact.
    private static let vocabulary: [Int: String] = [
        10: "Findings",
        11: " summary",
        20: "<end_of_turn>",
        21: "<start_of_turn>",
        22: "user",
        77: "<|im_end|>",
    ]

    private func tokenizer(eosTokenId: Int? = 99) -> StubStopTokenizer {
        StubStopTokenizer(vocabulary: Self.vocabulary, eosTokenId: eosTokenId)
    }

    private func stream(_ ids: [Int32]) -> AsyncStream<InferenceOutput> {
        AsyncStream { continuation in
            for id in ids { continuation.yield(InferenceOutput(tokenId: id)) }
            continuation.finish()
        }
    }

    private func responseText(_ events: [CoreAIStreamingOutputDecoder.Event]) -> String {
        events.reduce(into: "") { text, event in
            if case .response(let body) = event { text += body }
        }
    }

    private func run(
        tokens: [Int32],
        stopTokens: Set<Int32>
    ) async throws -> (
        events: [CoreAIStreamingOutputDecoder.Event],
        outcome: CoreAIVLMExecutor.GenerationOutcome
    ) {
        let collected = Collector()
        let outcome = try await CoreAIVLMExecutor.consumeGeneration(
            stream: stream(tokens),
            stopTokens: stopTokens,
            tokenizer: tokenizer(),
            profile: .gemma4Channels,
            reasoningEnabled: false
        ) { event in
            collected.append(event)
        }
        return (collected.events, outcome)
    }

    // MARK: - The stop set

    @Test("A declared end-of-turn token joins the stop set")
    func declaredEndOfTurnTokenIsAStopToken() {
        let stops = CoreAIVLMExecutor.stopTokenIDs(
            tokenizer: tokenizer(), additionalEosTokenIds: [20])
        #expect(stops == [99, 20])
    }

    @Test("An undeclared <|im_end|> is not assumed to end a turn")
    func undeclaredImEndIsNotAStopToken() {
        // The deleted line looked this spelling up in the vocabulary and added
        // it unconditionally. Restoring it puts 77 back in this set.
        let stops = CoreAIVLMExecutor.stopTokenIDs(
            tokenizer: tokenizer(), additionalEosTokenIds: [])
        #expect(stops == [99])
    }

    @Test("A bundle with no EOS at all still yields its declared stop tokens")
    func stopSetSurvivesAMissingEOS() {
        let stops = CoreAIVLMExecutor.stopTokenIDs(
            tokenizer: tokenizer(eosTokenId: nil), additionalEosTokenIds: [20])
        #expect(stops == [20])
    }

    // MARK: - The generation loop

    @Test("A declared end-of-turn token stops generation before it can leak")
    func declaredEndOfTurnTokenStopsGeneration() async throws {
        let stops = CoreAIVLMExecutor.stopTokenIDs(
            tokenizer: tokenizer(), additionalEosTokenIds: [20])
        // The model ends its turn (20) and then hallucinates a next turn.
        let (events, outcome) = try await run(
            tokens: [10, 11, 20, 21, 22], stopTokens: stops)

        #expect(responseText(events) == "Findings summary")
        #expect(!responseText(events).contains("<end_of_turn>"))
        #expect(!responseText(events).contains("<start_of_turn>"))
        #expect(outcome.stoppedOnStopToken)
        // Only the two tokens before the end of turn were generated content.
        #expect(outcome.generatedTokenCount == 2)
    }

    @Test("Without the declared token the same stream leaks the marker and a fabricated turn")
    func fixtureLeaksWhenTheStopTokenIsMissing() async throws {
        // Not a desired behavior — this pins that the fixture above is capable
        // of the failure, so the passing test is not vacuous. This is exactly
        // what the caller saw when the stop set guessed the model family.
        let stops = CoreAIVLMExecutor.stopTokenIDs(
            tokenizer: tokenizer(), additionalEosTokenIds: [])
        let (events, outcome) = try await run(
            tokens: [10, 11, 20, 21, 22], stopTokens: stops)

        #expect(responseText(events) == "Findings summary<end_of_turn><start_of_turn>user")
        #expect(!outcome.stoppedOnStopToken)
    }
}

/// Collects the events `consumeGeneration` streams out. The loop is `async`
/// but strictly sequential, so a plain reference box is enough.
private final class Collector: @unchecked Sendable {
    private(set) var events: [CoreAIStreamingOutputDecoder.Event] = []
    func append(_ event: CoreAIStreamingOutputDecoder.Event) { events.append(event) }
}

/// Minimal tokenizer over a fixed id→text vocabulary. `decode` concatenates,
/// which is what the executor's incremental detokenizer expects.
private struct StubStopTokenizer: Tokenizer, Sendable {
    let vocabulary: [Int: String]
    let eosTokenId: Int?

    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { eosTokenId.flatMap { vocabulary[$0] } }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }

    func tokenize(text: String) -> [String] { [text] }
    func encode(text: String) -> [Int] { [] }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func callAsFunction(_ text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokens: [Int]) -> String {
        tokens.map { vocabulary[$0] ?? "" }.joined()
    }
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String { decode(tokens: tokens) }
    func convertTokenToId(_ token: String) -> Int? {
        vocabulary.first { $0.value == token }?.key
    }
    func convertTokensToIds(_ tokens: [String]) -> [Int?] { tokens.map(convertTokenToId) }
    func convertIdToToken(_ id: Int) -> String? { vocabulary[id] }
    func convertIdsToTokens(_ ids: [Int]) -> [String?] { ids.map(convertIdToToken) }

    func applyChatTemplate(messages: [Message]) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] { [] }
    func applyChatTemplate(
        messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] {
        []
    }
    func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] { [] }
    func applyChatTemplate(
        messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool,
        truncation: Bool, maxLength: Int?, tools: [ToolSpec]?
    ) throws -> [Int] { [] }
    func applyChatTemplate(
        messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool,
        truncation: Bool, maxLength: Int?, tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

#endif
