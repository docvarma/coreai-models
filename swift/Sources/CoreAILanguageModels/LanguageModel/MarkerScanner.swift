// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Incremental text buffer that never emits a partial reserved marker.
///
/// Generalizes the single-marker hold-back that `ThinkTagParser` and
/// `ToolCallParser` each implemented separately: callers supply the set of
/// markers currently meaningful, and the scanner withholds only the trailing
/// characters that could still grow into one of them.
package struct MarkerScanner {
    private var buffer: String = ""

    package init() {}

    package var isEmpty: Bool { buffer.isEmpty }

    package mutating func append(_ delta: String) { buffer.append(delta) }

    /// Earliest complete marker in the buffer. At an equal start index the
    /// longest marker wins, so `<|channel|>` is preferred over `<|channel`.
    package func firstMatch(
        of markers: [String]
    ) -> (marker: String, range: Range<String.Index>)? {
        var best: (marker: String, range: Range<String.Index>)?
        for marker in markers {
            guard !marker.isEmpty, let range = buffer.range(of: marker) else { continue }
            guard let current = best else {
                best = (marker, range)
                continue
            }
            let earlier = range.lowerBound < current.range.lowerBound
            let longerAtSameStart =
                range.lowerBound == current.range.lowerBound
                && marker.count > current.marker.count
            if earlier || longerAtSameStart { best = (marker, range) }
        }
        return best
    }

    /// Earliest match that is safe to act on, i.e. one that no longer marker
    /// can still grow out of once more text arrives.
    ///
    /// `firstMatch` alone is not enough when one marker is a prefix of another:
    /// a buffer ending in `<|tool_call>` matches that marker exactly, yet the
    /// next delta may turn it into `<|tool_call>call:`. Acting on the shorter
    /// spelling there would reject a well-formed call. Withholding is bounded —
    /// `takeSafe` holds the same bytes back, and `isFinal` settles everything.
    package func firstSettledMatch(
        of markers: [String],
        isFinal: Bool
    ) -> (marker: String, range: Range<String.Index>)? {
        guard let best = firstMatch(of: markers) else { return nil }
        guard !isFinal else { return best }
        let tail = buffer[best.range.lowerBound...]
        let canStillGrow = markers.contains { $0.count > tail.count && $0.starts(with: tail) }
        return canStillGrow ? nil : best
    }

    /// Removes and returns the text that cannot be part of a pending marker.
    /// When `isFinal` is true nothing is held back.
    package mutating func takeSafe(waitingFor markers: [String], isFinal: Bool) -> String {
        let safe = isFinal ? buffer.endIndex : lastSafeIndex(forAny: markers)
        guard safe > buffer.startIndex else { return "" }
        let text = String(buffer[buffer.startIndex..<safe])
        buffer = String(buffer[safe...])
        return text
    }

    /// Removes everything through `range`, returning the text that preceded it.
    package mutating func takeUpTo(_ range: Range<String.Index>) -> String {
        let before = String(buffer[buffer.startIndex..<range.lowerBound])
        buffer = String(buffer[range.upperBound...])
        return before
    }

    package mutating func takeAll() -> String {
        defer { buffer.removeAll(keepingCapacity: false) }
        return buffer
    }

    /// Rightmost index whose suffix is not a non-empty prefix of any marker.
    /// Scans at most `longestMarker - 1` trailing characters.
    private func lastSafeIndex(forAny markers: [String]) -> String.Index {
        let maxHold = (markers.map(\.count).max() ?? 1) - 1
        guard !buffer.isEmpty, maxHold > 0 else { return buffer.endIndex }
        let holdStart = buffer.index(buffer.endIndex, offsetBy: -min(maxHold, buffer.count))
        for offset in 0..<buffer.distance(from: holdStart, to: buffer.endIndex) {
            let index = buffer.index(holdStart, offsetBy: offset)
            let suffix = buffer[index...]
            if markers.contains(where: { $0.starts(with: suffix) }) { return index }
        }
        return buffer.endIndex
    }
}
