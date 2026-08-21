// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

@Suite("MarkerScanner")
struct MarkerScannerTests {
    @Test("Emits everything when no marker can match")
    func emitsWhenNoPartialMatch() {
        var scanner = MarkerScanner()
        scanner.append("Hello, world!")
        #expect(scanner.takeSafe(waitingFor: ["<think>"], isFinal: false) == "Hello, world!")
    }

    @Test("Holds back a partial marker suffix")
    func holdsBackPartialSuffix() {
        var scanner = MarkerScanner()
        scanner.append("before<thi")
        #expect(scanner.takeSafe(waitingFor: ["<think>"], isFinal: false) == "before")
        scanner.append("nk>")
        let match = scanner.firstMatch(of: ["<think>"])
        #expect(match?.marker == "<think>")
    }

    @Test("Final flush emits the held-back suffix")
    func finalFlushEmitsHeldBack() {
        var scanner = MarkerScanner()
        scanner.append("trailing<thi")
        _ = scanner.takeSafe(waitingFor: ["<think>"], isFinal: false)
        #expect(scanner.takeSafe(waitingFor: ["<think>"], isFinal: true) == "<thi")
    }

    @Test("A complete marker that a longer marker can still grow from is unsettled")
    func settledMatchWaitsForLongerMarker() {
        var scanner = MarkerScanner()
        scanner.append("<|tool_call>")
        let markers = ["<|tool_call>", "<|tool_call>call:"]
        #expect(scanner.firstMatch(of: markers)?.marker == "<|tool_call>")
        #expect(scanner.firstSettledMatch(of: markers, isFinal: false) == nil)
        #expect(scanner.firstSettledMatch(of: markers, isFinal: true)?.marker == "<|tool_call>")
    }

    @Test("A settled match is returned once the longer marker is ruled out")
    func settledMatchOnceLongerMarkerRuledOut() {
        var scanner = MarkerScanner()
        scanner.append("<|tool_call>")
        let markers = ["<|tool_call>", "<|tool_call>call:"]
        #expect(scanner.firstSettledMatch(of: markers, isFinal: false) == nil)
        scanner.append("oops")
        #expect(scanner.firstSettledMatch(of: markers, isFinal: false)?.marker == "<|tool_call>")
        scanner = MarkerScanner()
        scanner.append("<|tool_call>call:x")
        #expect(
            scanner.firstSettledMatch(of: markers, isFinal: false)?.marker == "<|tool_call>call:")
    }

    @Test("Holds back against the longest candidate marker")
    func holdsBackAgainstLongestMarker() {
        var scanner = MarkerScanner()
        scanner.append("text<atem:function")
        // Must hold the whole partial, not just the length of the shortest marker.
        #expect(scanner.takeSafe(waitingFor: ["<|end|>", "<atem:function_calls>"], isFinal: false) == "text")
    }

    @Test("Earliest marker wins across candidates")
    func earliestMarkerWins() {
        var scanner = MarkerScanner()
        scanner.append("a<|end|>b<|call|>c")
        let match = scanner.firstMatch(of: ["<|call|>", "<|end|>"])
        #expect(match?.marker == "<|end|>")
    }

    @Test("Longest marker wins at an equal start index")
    func longestMarkerWinsAtSameStart() {
        var scanner = MarkerScanner()
        scanner.append("x<|channel|>analysis")
        let match = scanner.firstMatch(of: ["<|channel", "<|channel|>"])
        #expect(match?.marker == "<|channel|>")
    }

    @Test("takeUpTo returns preceding text and consumes the marker")
    func takeUpToConsumesMarker() {
        var scanner = MarkerScanner()
        scanner.append("before<think>after")
        let match = scanner.firstMatch(of: ["<think>"])
        let before = try! #require(match).range
        #expect(scanner.takeUpTo(before) == "before")
        #expect(scanner.takeAll() == "after")
    }

    @Test("Empty marker in the set is ignored, never matched")
    func emptyMarkerIgnored() {
        var scanner = MarkerScanner()
        scanner.append("plain text")
        #expect(scanner.firstMatch(of: ["", "<think>"]) == nil)
        #expect(scanner.takeSafe(waitingFor: ["", "<think>"], isFinal: false) == "plain text")
    }

    @Test("Marker longer than the buffer holds the whole buffer back")
    func markerLongerThanBuffer() {
        var scanner = MarkerScanner()
        scanner.append("<a")
        #expect(scanner.takeSafe(waitingFor: ["<atem:function_calls>"], isFinal: false) == "")
        #expect(scanner.takeSafe(waitingFor: ["<atem:function_calls>"], isFinal: true) == "<a")
    }

    @Test("Hold-back window never splits a grapheme cluster")
    func holdBackDoesNotSplitGrapheme() {
        var scanner = MarkerScanner()
        // The family emoji is one Character but many scalars. "<thi" is a live
        // partial of "<think>", so the hold-back boundary lands immediately
        // after the cluster — the split point the Character-based window must
        // get right.
        scanner.append("ok 👩‍👩‍👧‍👦<thi")
        let emitted = scanner.takeSafe(waitingFor: ["<think>"], isFinal: false)
        #expect(emitted == "ok 👩‍👩‍👧‍👦", "hold-back must release the cluster intact and withhold only the partial marker")

        scanner.append("nk>after")
        let match = scanner.firstMatch(of: ["<think>"])
        #expect(match?.marker == "<think>")
    }
}

#endif
