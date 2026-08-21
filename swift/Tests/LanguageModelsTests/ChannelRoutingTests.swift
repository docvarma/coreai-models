// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

/// The decoder-to-channel mapping is the integration point this whole plan
/// converges on, and `LanguageModelExecutorGenerationChannel.Event` is an
/// opaque struct with no readable properties — a test can never inspect what
/// was sent. `CoreAIChannelRouting` is the decision that mapping makes, split
/// out as plain data precisely so it can be asserted; `dispatch` is then three
/// unconditional sends over it.
@Suite("Channel routing")
struct ChannelRoutingTests {
    @Test("Response text routes to the response channel arm")
    func responseRoutesToResponse() {
        #expect(
            CoreAIChannelRouting(.response("synthetic-answer"))
                == .appendResponseText("synthetic-answer"))
    }

    @Test("Reasoning text routes to the reasoning channel arm, not into the response")
    func reasoningRoutesToReasoning() {
        // Swapping these two arms is the mistake that leaks chain-of-thought
        // into the user-facing response entry.
        #expect(
            CoreAIChannelRouting(.reasoning("synthetic-thought"))
                == .appendReasoningText("synthetic-thought"))
    }

    @Test("A tool call routes to the tool-calls arm with its id, name and arguments intact")
    func toolCallRoutesToToolCalls() {
        let routing = CoreAIChannelRouting(
            .toolCall(id: "synthetic-id", name: "synthetic.tool", argumentsJSON: "{\"value\":1}"))
        #expect(
            routing
                == .appendToolCallArguments(
                    id: "synthetic-id",
                    name: "synthetic.tool",
                    argumentsJSON: "{\"value\":1}"))
    }

    @Test("An empty response fragment is dropped rather than sent")
    func emptyResponseIsDropped() {
        // The superseded parsers guarded `if !text.isEmpty` before sending;
        // dropping that guard sends empty appendText events to the caller.
        #expect(CoreAIChannelRouting(.response("")) == .drop)
    }

    @Test("An empty reasoning fragment is dropped rather than sent")
    func emptyReasoningIsDropped() {
        #expect(CoreAIChannelRouting(.reasoning("")) == .drop)
    }

    @Test("A tool call with empty arguments is still dispatched")
    func emptyToolArgumentsAreStillDispatched() {
        // Suppression applies to text fragments only: a zero-argument call is
        // a real call and must reach the session.
        #expect(
            CoreAIChannelRouting(
                .toolCall(id: "synthetic-id", name: "synthetic.tool", argumentsJSON: ""))
                == .appendToolCallArguments(
                    id: "synthetic-id", name: "synthetic.tool", argumentsJSON: ""))
    }
}

#endif
