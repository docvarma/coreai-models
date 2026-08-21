// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

@Suite("Envelope decoder")
struct EnvelopeDecoderTests {
    private func drain(
        profile: CoreAILanguageProtocolProfile,
        reasoningEnabled: Bool,
        deltas: [String]
    ) throws -> [CoreAIStreamingOutputDecoder.Event] {
        var decoder = CoreAIStreamingOutputDecoder(
            profile: profile, reasoningEnabled: reasoningEnabled)
        var events: [CoreAIStreamingOutputDecoder.Event] = []
        for delta in deltas { events += try decoder.consume(delta) }
        return events + (try decoder.finish(truncated: false))
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

    private let harmonyWhole =
        "<|start|>assistant<|channel|>analysis<|message|>weighing<|end|>"
        + "<|start|>assistant<|channel|>final<|message|>the answer<|return|>"

    /// The recipient precedes the channel in a harmony tool header; that is
    /// the spelling `parseHarmony` matched before this decoder replaced it.
    private let harmonyToolWhole =
        "<|start|>assistant to=functions.synthetic.tool<|channel|>commentary"
        + "<|message|>{\"value\":1}<|call|>"

    private let atemWhole =
        "<|start|>assistant to=self<|message|>thinking<|eom|>"
        + "<|start|>assistant to=user<|message|>done<|eot|>"

    @Test("Final-channel body streams before its terminator arrives")
    func finalBodyStreamsEarly() throws {
        var decoder = CoreAIStreamingOutputDecoder(profile: .harmony, reasoningEnabled: true)
        _ = try decoder.consume("<|start|>assistant<|channel|>analysis<|message|>weighing<|end|>")
        _ = try decoder.consume("<|start|>assistant<|channel|>final<|message|>")
        let events = try decoder.consume("the ans")
        #expect(events == [.response("the ans")])
    }

    @Test("ATEM user body streams before its terminator arrives")
    func atemUserBodyStreamsEarly() throws {
        var decoder = CoreAIStreamingOutputDecoder(profile: .atem, reasoningEnabled: true)
        _ = try decoder.consume("<|start|>assistant to=self<|message|>thinking<|eom|>")
        _ = try decoder.consume("<|start|>assistant to=user<|message|>")
        let events = try decoder.consume("do")
        #expect(events == [.response("do")], "the body must not wait for <|eot|>")
    }

    @Test("Analysis channel routes to reasoning")
    func analysisRoutesToReasoning() throws {
        let events = try drain(profile: .harmony, reasoningEnabled: true, deltas: [harmonyWhole])
        #expect(events == [.reasoning("weighing"), .response("the answer")])
    }

    @Test("Marker split across every interior offset yields identical events")
    func markerSplitAtEveryOffset() throws {
        let expected = merged(
            try drain(profile: .harmony, reasoningEnabled: true, deltas: [harmonyWhole]))
        #expect(expected == [.reasoning("weighing"), .response("the answer")])
        var offsetsChecked = 0
        for offset in 1..<harmonyWhole.count {
            let index = harmonyWhole.index(harmonyWhole.startIndex, offsetBy: offset)
            let parts = [String(harmonyWhole[..<index]), String(harmonyWhole[index...])]
            let actual = merged(
                try drain(profile: .harmony, reasoningEnabled: true, deltas: parts))
            #expect(actual == expected, "split at \(offset) diverged")
            offsetsChecked += 1
        }
        #expect(offsetsChecked == harmonyWhole.count - 1)
    }

    @Test("ATEM markers split across every interior offset yield identical events")
    func atemMarkerSplitAtEveryOffset() throws {
        let expected = merged(
            try drain(profile: .atem, reasoningEnabled: true, deltas: [atemWhole]))
        #expect(expected == [.reasoning("thinking"), .response("done")])
        var offsetsChecked = 0
        for offset in 1..<atemWhole.count {
            let index = atemWhole.index(atemWhole.startIndex, offsetBy: offset)
            let parts = [String(atemWhole[..<index]), String(atemWhole[index...])]
            let actual = merged(try drain(profile: .atem, reasoningEnabled: true, deltas: parts))
            #expect(actual == expected, "split at \(offset) diverged")
            offsetsChecked += 1
        }
        #expect(offsetsChecked == atemWhole.count - 1)
    }

    @Test("Unterminated envelope fails at finish")
    func unterminatedEnvelopeFails() {
        #expect(
            throws: CoreAIProtocolError(profile: .harmony, failure: .malformedChannel)
        ) {
            try drain(
                profile: .harmony, reasoningEnabled: true,
                deltas: ["<|start|>assistant<|channel|>final<|message|>dangling"])
        }
    }

    @Test("Unterminated header fails at finish")
    func unterminatedHeaderFails() {
        #expect(
            throws: CoreAIProtocolError(profile: .harmony, failure: .malformedChannel)
        ) {
            try drain(
                profile: .harmony, reasoningEnabled: true,
                deltas: ["<|start|>assistant<|channel|>fin"])
        }
    }

    @Test("Tool recipient buffers until the envelope terminates")
    func toolRecipientBuffers() throws {
        var decoder = CoreAIStreamingOutputDecoder(profile: .harmony, reasoningEnabled: false)
        let partial = try decoder.consume(
            "<|start|>assistant to=functions.synthetic.tool<|channel|>commentary"
                + "<|message|>{\"value\":1")
        #expect(partial.isEmpty, "a partial tool call must not be dispatched")
    }

    @Test("Harmony tool envelope dispatches at its terminator")
    func harmonyToolCallEmittedWhenComplete() throws {
        let events = try drain(
            profile: .harmony, reasoningEnabled: false, deltas: [harmonyToolWhole])
        #expect(
            events == [
                .toolCall(
                    id: "coreai-call-1", name: "synthetic.tool", argumentsJSON: "{\"value\":1}")
            ])
    }

    @Test("A tool body split across deltas reassembles into one call")
    func toolBodySplitAcrossDeltasReassembles() throws {
        var decoder = CoreAIStreamingOutputDecoder(profile: .harmony, reasoningEnabled: false)
        var events = try decoder.consume(
            "<|start|>assistant to=functions.synthetic.tool<|channel|>commentary<|message|>")
        events += try decoder.consume("{\"value\"")
        events += try decoder.consume(":1")
        #expect(events.isEmpty, "no fragment of the body may be dispatched")
        events += try decoder.consume("}<|call|>")
        events += try decoder.finish(truncated: false)
        #expect(
            events == [
                .toolCall(
                    id: "coreai-call-1", name: "synthetic.tool", argumentsJSON: "{\"value\":1}")
            ])
    }

    @Test("ATEM tool envelope dispatches its markup calls at the terminator")
    func atemToolCallEmittedWhenComplete() throws {
        let whole =
            "<|start|>assistant to=functions<|message|><atem:function_calls>"
            + "<atem:invoke name=\"synthetic.tool\">"
            + "<atem:parameter name=\"value\">1</atem:parameter>"
            + "</atem:invoke></atem:function_calls><|eom|>"
        let events = try drain(profile: .atem, reasoningEnabled: false, deltas: [whole])
        #expect(
            events == [
                .toolCall(
                    id: "coreai-call-1", name: "synthetic.tool", argumentsJSON: "{\"value\":1}")
            ])
    }

    @Test("ATEM self recipient routes to reasoning")
    func atemSelfRoutesToReasoning() throws {
        let events = try drain(profile: .atem, reasoningEnabled: true, deltas: [atemWhole])
        #expect(events.contains(.reasoning("thinking")))
        #expect(events.contains(.response("done")))
    }

    @Test("Reasoning envelope while reasoning is disabled fails")
    func reasoningWhileDisabledFails() {
        for testCase in [
            (CoreAILanguageProtocolProfile.harmony, harmonyWhole),
            (CoreAILanguageProtocolProfile.atem, atemWhole),
        ] {
            #expect(
                throws: CoreAIProtocolError(profile: testCase.0, failure: .malformedReasoning)
            ) {
                try drain(profile: testCase.0, reasoningEnabled: false, deltas: [testCase.1])
            }
        }
    }

    @Test("An unknown channel never reaches the caller as response text")
    func unknownChannelFails() {
        #expect(
            throws: CoreAIProtocolError(profile: .harmony, failure: .malformedChannel)
        ) {
            try drain(
                profile: .harmony, reasoningEnabled: true,
                deltas: ["<|start|>assistant<|channel|>mystery<|message|>leak<|end|>"])
        }
    }

    @Test("An envelope missing its start marker is rejected")
    func headerWithoutStartMarkerFails() {
        #expect(
            throws: CoreAIProtocolError(profile: .harmony, failure: .malformedChannel)
        ) {
            try drain(
                profile: .harmony, reasoningEnabled: true,
                deltas: ["loose text<|message|>leak<|return|>"])
        }
    }

    @Test("Channel terminated by the wrong marker is rejected")
    func wrongTerminatorFails() {
        #expect(
            throws: CoreAIProtocolError(profile: .harmony, failure: .malformedChannel)
        ) {
            try drain(
                profile: .harmony, reasoningEnabled: true,
                deltas: ["<|start|>assistant<|channel|>analysis<|message|>weighing<|return|>"])
        }
    }

    @Test("A second response envelope is a duplicate")
    func duplicateResponseEnvelopeFails() {
        #expect(
            throws: CoreAIProtocolError(profile: .harmony, failure: .duplicateProtocolBlock)
        ) {
            try drain(
                profile: .harmony, reasoningEnabled: true,
                deltas: [
                    harmonyWhole
                        + "<|start|>assistant<|channel|>final<|message|>again<|return|>"
                ])
        }
    }

    @Test("A tool envelope after the response envelope is rejected")
    func toolAfterResponseFails() {
        #expect(
            throws: CoreAIProtocolError(profile: .harmony, failure: .mixedResponseAndToolCall)
        ) {
            try drain(
                profile: .harmony, reasoningEnabled: true,
                deltas: [harmonyWhole + harmonyToolWhole])
        }
    }
}

#endif
