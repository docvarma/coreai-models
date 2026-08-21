// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

@Suite("Inline decoder")
struct InlineDecoderTests {
    private func drain(
        profile: CoreAILanguageProtocolProfile,
        reasoningEnabled: Bool,
        deltas: [String]
    ) throws -> [CoreAIStreamingOutputDecoder.Event] {
        var decoder = CoreAIStreamingOutputDecoder(
            profile: profile, reasoningEnabled: reasoningEnabled)
        var events: [CoreAIStreamingOutputDecoder.Event] = []
        for delta in deltas { events += try decoder.consume(delta) }
        return events + (try decoder.finish())
    }

    /// Chunking decides how many `.response`/`.reasoning` events a run of text
    /// arrives in, so the chunk-invariant property is over the *content*
    /// stream: adjacent same-kind text events are one logical run.
    private func merged(
        _ events: [CoreAIStreamingOutputDecoder.Event]
    ) -> [CoreAIStreamingOutputDecoder.Event] {
        var result: [CoreAIStreamingOutputDecoder.Event] = []
        for event in events {
            switch (result.last, event) {
            case (.response(let previous), .response(let next)):
                result[result.count - 1] = .response(previous + next)
            case (.reasoning(let previous), .reasoning(let next)):
                result[result.count - 1] = .reasoning(previous + next)
            default:
                result.append(event)
            }
        }
        return result
    }

    @Test("Response text is emitted before the stream ends")
    func responseStreamsBeforeFinish() throws {
        var decoder = CoreAIStreamingOutputDecoder(
            profile: .plainChat, reasoningEnabled: false)
        let events = try decoder.consume("partial answer")
        #expect(events == [.response("partial answer")])
    }

    @Test("Reasoning body streams before its block closes")
    func reasoningStreamsBeforeClose() throws {
        var decoder = CoreAIStreamingOutputDecoder(
            profile: .qwen35XML, reasoningEnabled: true)
        _ = try decoder.consume("<think>")
        let events = try decoder.consume("stepping through")
        #expect(events == [.reasoning("stepping through")])
    }

    @Test("Marker split across every interior offset yields identical events")
    func markerSplitAtEveryOffset() throws {
        let whole = "<think>why</think>answer"
        let expected = merged(
            try drain(profile: .qwen35XML, reasoningEnabled: true, deltas: [whole]))
        #expect(expected == [.reasoning("why"), .response("answer")])
        var offsetsChecked = 0
        for offset in 1..<whole.count {
            let index = whole.index(whole.startIndex, offsetBy: offset)
            let parts = [String(whole[..<index]), String(whole[index...])]
            let actual = merged(
                try drain(profile: .qwen35XML, reasoningEnabled: true, deltas: parts))
            #expect(actual == expected, "split at \(offset) diverged")
            offsetsChecked += 1
        }
        #expect(offsetsChecked == whole.count - 1)
    }

    @Test("Absent reasoning is valid when reasoning is enabled")
    func absentReasoningIsValid() throws {
        let events = try drain(
            profile: .qwen35XML, reasoningEnabled: true, deltas: ["a short answer"])
        #expect(events == [.response("a short answer")])
    }

    @Test("Unclosed reasoning block fails at finish")
    func unclosedReasoningFails() {
        #expect(throws: CoreAIProtocolError.self) {
            try drain(profile: .qwen35XML, reasoningEnabled: true, deltas: ["<think>never closed"])
        }
    }

    @Test("Reasoning while disabled is a failure")
    func reasoningWhileDisabledFails() {
        #expect(throws: CoreAIProtocolError.self) {
            try drain(profile: .qwen35XML, reasoningEnabled: false, deltas: ["<think>x</think>"])
        }
    }

    @Test("plainChat rejects any reserved marker")
    func plainChatRejectsReservedMarker() {
        #expect(throws: CoreAIProtocolError.self) {
            try drain(profile: .plainChat, reasoningEnabled: false, deltas: ["ok <think> leak"])
        }
    }

    @Test("plainChat rejects a reserved marker split across deltas")
    func plainChatRejectsSplitReservedMarker() {
        #expect(throws: CoreAIProtocolError.self) {
            try drain(profile: .plainChat, reasoningEnabled: false, deltas: ["ok <thi", "nk> leak"])
        }
    }

    @Test("Tool call is not emitted until structurally complete")
    func toolCallBuffersUntilComplete() throws {
        var decoder = CoreAIStreamingOutputDecoder(
            profile: .qwen35XML, reasoningEnabled: false)
        let partial = try decoder.consume("<tool_call><function=synthetic.tool><parameter=value>1")
        #expect(partial.isEmpty, "a partial tool call must not be dispatched")
    }

    @Test("A top-level marker of the selected profile never reaches the caller")
    func misplacedProfileMarkerFails() {
        // Each case is split mid-marker so hold-back has to reassemble it
        // before the violation can be seen at all.
        let cases:
            [(
                profile: CoreAILanguageProtocolProfile, reasoning: Bool, deltas: [String],
                failure: CoreAIProtocolFailure
            )] = [
                (.gemma4Channels, false, ["answer <|tool_", "call>oops"], .malformedToolCall),
                (.gemma4Channels, false, ["answer <tool_c", "all|> more"], .malformedToolCall),
                (.gemma4Channels, false, ["answer <|chan", "nel>final"], .malformedChannel),
                (.qwen35XML, false, ["answer </tool_", "call> more"], .malformedToolCall),
                (.qwen35XML, true, ["<think>why</think>ok </thi", "nk> more"], .duplicateProtocolBlock),
            ]
        for testCase in cases {
            #expect(
                throws: CoreAIProtocolError(
                    profile: testCase.profile, failure: testCase.failure)
            ) {
                try drain(
                    profile: testCase.profile, reasoningEnabled: testCase.reasoning,
                    deltas: testCase.deltas)
            }
        }
    }

    @Test("A profile marker misplaced inside reasoning fails too")
    func misplacedMarkerInsideReasoningFails() {
        #expect(
            throws: CoreAIProtocolError(profile: .qwen35XML, failure: .malformedToolCall)
        ) {
            try drain(
                profile: .qwen35XML, reasoningEnabled: true,
                deltas: ["<think>weighing </tool_", "call> options</think>ok"])
        }
    }

    @Test("A looser delimiter spelling is held back until its longer form settles")
    func longerDelimiterSpellingWins() throws {
        // `<|tool_call>` alone is a violation, but it is also a prefix of the
        // real opener, so it must not be acted on until the next delta lands.
        var decoder = CoreAIStreamingOutputDecoder(
            profile: .gemma4Channels, reasoningEnabled: false)
        let held = try decoder.consume("<|tool_call>")
        #expect(held.isEmpty)
        let events =
            try decoder.consume("call:synthetic.tool{value: 1}<tool_call|>")
            + (try decoder.finish())
        #expect(
            events == [
                .toolCall(
                    id: "coreai-call-1", name: "synthetic.tool", argumentsJSON: "{\"value\":1}")
            ])
    }

    @Test("Complete qwen tool block is emitted at its closing marker")
    func qwenToolCallEmittedWhenComplete() throws {
        let events = try drain(
            profile: .qwen35XML, reasoningEnabled: false,
            deltas: [
                "<tool_call><function=synthetic.tool>",
                "<parameter=value>1</parameter></function></tool_call>",
            ])
        #expect(
            events == [
                .toolCall(
                    id: "coreai-call-1", name: "synthetic.tool", argumentsJSON: "{\"value\":1}")
            ])
    }

    @Test("Gemma reasoning and tool blocks stream inline")
    func gemmaInlineBlocks() throws {
        let events = try drain(
            profile: .gemma4Channels, reasoningEnabled: true,
            deltas: [
                "<|channel>thought\nsynthetic-reasoning<chan",
                "nel|><|tool_call>call:synthetic.tool{value: 1}<tool_call|>",
            ])
        #expect(
            events == [
                .reasoning("synthetic-reasoning"),
                .toolCall(
                    id: "coreai-call-1", name: "synthetic.tool", argumentsJSON: "{\"value\":1}"),
            ])
    }

    @Test("Envelope profiles still decode at finish")
    func envelopeProfilesDecodeAtFinish() throws {
        var decoder = CoreAIStreamingOutputDecoder(profile: .harmony, reasoningEnabled: true)
        let early = try decoder.consume("<|start|>assistant<|channel|>analysis<|message|>")
        #expect(early.isEmpty)
        _ = try decoder.consume("synthetic-reasoning<|end|>")
        _ = try decoder.consume("<|start|>assistant<|channel|>final<|message|>")
        let events = try decoder.consume("synthetic-response<|return|>") + (try decoder.finish())
        #expect(events == [.reasoning("synthetic-reasoning"), .response("synthetic-response")])
    }
}

#endif
