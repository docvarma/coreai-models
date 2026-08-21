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
        // The family emoji is one Character but many scalars. The hold-back
        // window is measured in Characters and must not slice through it.
        scanner.append("ok 👩‍👩‍👧‍👦")
        let emitted = scanner.takeSafe(waitingFor: ["<think>"], isFinal: false)
        let rest = scanner.takeSafe(waitingFor: ["<think>"], isFinal: true)
        #expect(emitted + rest == "ok 👩‍👩‍👧‍👦")
    }
}

#endif
