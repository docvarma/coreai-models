// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

/// An unterminated block at the end of a stream means one of two things, and
/// only the caller knows which: the model chose to stop mid-block (a protocol
/// violation), or we stopped it — the token cap, a cancellation, an engine
/// error (ordinary truncation). `finish(truncated:)` carries that distinction.
///
/// Every fixture below ends on a lone `<`, which is a live prefix of both
/// `</think>` and `<|end|>`, so the scanner is genuinely holding it back when
/// the stream ends. Without that the drain would already have emitted
/// everything and the two paths would be indistinguishable.
@Suite("Decoder truncation")
struct DecoderTruncationTests {
    private func reasoningText(_ events: [CoreAIStreamingOutputDecoder.Event]) -> String {
        events.reduce(into: "") { text, event in
            if case .reasoning(let body) = event { text += body }
        }
    }

    private func responseText(_ events: [CoreAIStreamingOutputDecoder.Event]) -> String {
        events.reduce(into: "") { text, event in
            if case .response(let body) = event { text += body }
        }
    }

    // MARK: - Inline profiles

    @Test("A capped generation flushes its partial reasoning instead of failing")
    func inlineTruncationFlushesOpenReasoning() throws {
        var decoder = CoreAIStreamingOutputDecoder(profile: .qwen35XML, reasoningEnabled: true)
        var events = try decoder.consume("synthetic-answer<think>synthetic-thought<")
        events += try decoder.finish(truncated: true)
        // Both halves survive. Before this fix the throw discarded the whole
        // response after its events had already been streamed to the caller.
        #expect(responseText(events) == "synthetic-answer")
        #expect(reasoningText(events) == "synthetic-thought<")
    }

    @Test("The same open reasoning block at end-of-turn is still a violation")
    func inlineEndOfTurnOnOpenReasoningStillThrows() throws {
        var decoder = CoreAIStreamingOutputDecoder(profile: .qwen35XML, reasoningEnabled: true)
        _ = try decoder.consume("synthetic-answer<think>synthetic-thought<")
        #expect(throws: CoreAIProtocolError(profile: .qwen35XML, failure: .unfinishedReasoning)) {
            _ = try decoder.finish(truncated: false)
        }
    }

    @Test("A capped generation drops a half-buffered tool call rather than dispatching it")
    func inlineTruncationDropsPartialToolCall() throws {
        var decoder = CoreAIStreamingOutputDecoder(profile: .qwen35XML, reasoningEnabled: false)
        var events = try decoder.consume("synthetic-answer<tool_call><function=synthetic.tool>")
        events += try decoder.finish(truncated: true)
        #expect(responseText(events) == "synthetic-answer")
        #expect(
            !events.contains {
                if case .toolCall = $0 { return true }
                return false
            }, "half a tool call must never reach the caller")
    }

    @Test("The same open tool block at end-of-turn is still a violation")
    func inlineEndOfTurnOnOpenToolCallStillThrows() throws {
        var decoder = CoreAIStreamingOutputDecoder(profile: .qwen35XML, reasoningEnabled: false)
        _ = try decoder.consume("synthetic-answer<tool_call><function=synthetic.tool>")
        #expect(throws: CoreAIProtocolError(profile: .qwen35XML, failure: .unfinishedToolCall)) {
            _ = try decoder.finish(truncated: false)
        }
    }

    // MARK: - Envelope profiles

    @Test("A capped envelope flushes its partial body instead of failing")
    func envelopeTruncationFlushesOpenBody() throws {
        var decoder = CoreAIStreamingOutputDecoder(profile: .harmony, reasoningEnabled: true)
        var events = try decoder.consume(
            "<|start|>assistant<|channel|>analysis<|message|>synthetic-thought<")
        events += try decoder.finish(truncated: true)
        #expect(reasoningText(events) == "synthetic-thought<")
    }

    @Test("The same unterminated envelope at end-of-turn is still a violation")
    func envelopeEndOfTurnOnOpenBodyStillThrows() throws {
        var decoder = CoreAIStreamingOutputDecoder(profile: .harmony, reasoningEnabled: true)
        _ = try decoder.consume(
            "<|start|>assistant<|channel|>analysis<|message|>synthetic-thought<")
        #expect(throws: CoreAIProtocolError(profile: .harmony, failure: .unfinishedReasoning)) {
            _ = try decoder.finish(truncated: false)
        }
    }

    @Test("A capped generation accepts a half-written envelope header")
    func envelopeTruncationAcceptsPartialHeader() throws {
        var decoder = CoreAIStreamingOutputDecoder(profile: .harmony, reasoningEnabled: true)
        var events = try decoder.consume("<|start|>assistant<|channel|>ana")
        events += try decoder.finish(truncated: true)
        // A header is protocol, never content, so nothing is emitted — but
        // being cut off mid-header is not a violation either.
        #expect(events.isEmpty)
    }

    @Test("The same half-written header at end-of-turn is still a violation")
    func envelopeEndOfTurnOnPartialHeaderStillThrows() throws {
        var decoder = CoreAIStreamingOutputDecoder(profile: .harmony, reasoningEnabled: true)
        _ = try decoder.consume("<|start|>assistant<|channel|>ana")
        #expect(throws: CoreAIProtocolError(profile: .harmony, failure: .malformedChannel)) {
            _ = try decoder.finish(truncated: false)
        }
    }

    // MARK: - Truncation does not weaken in-stream detection

    // MARK: - Which stop reasons count as truncation

    @Test("Only an end-of-sequence token counts as the model choosing to stop")
    func onlyEOSIsAModelChosenStop() {
        #expect(!CoreAILanguageModel.CoreAIExecutor.isTruncated(.eos))
        // Everything else is our cut, not the model's end-of-turn.
        #expect(CoreAILanguageModel.CoreAIExecutor.isTruncated(.maxTokens))
        #expect(CoreAILanguageModel.CoreAIExecutor.isTruncated(.cancelled))
        #expect(CoreAILanguageModel.CoreAIExecutor.isTruncated(.error))
        #expect(CoreAILanguageModel.CoreAIExecutor.isTruncated(.stopSequence("synthetic-stop")))
        #expect(CoreAILanguageModel.CoreAIExecutor.isTruncated(nil))
    }

    @Test("Truncation does not excuse a violation the stream already committed")
    func truncationDoesNotExcuseAnInStreamViolation() throws {
        var decoder = CoreAIStreamingOutputDecoder(profile: .qwen35XML, reasoningEnabled: true)
        // A second reasoning block is rejected while consuming, long before
        // any end-of-stream decision — `truncated:` must not reach it.
        #expect(throws: CoreAIProtocolError(profile: .qwen35XML, failure: .duplicateProtocolBlock)) {
            _ = try decoder.consume("<think>one</think>text<think>two")
        }
    }
}

#endif
