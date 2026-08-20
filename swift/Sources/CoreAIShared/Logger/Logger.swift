// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Synchronization

/// Centralized logging utility used across engines, tokenizers, decoding strategies,
/// sampling strategies, and command-line tools.
public struct CLILogger {
    public typealias Sink = @Sendable (_ level: Int, _ component: String?, _ message: String) -> Void

    private struct State: Sendable {
        var level = 0
        var sink: Sink?
    }

    private static let state = Mutex(State())

    public static var level: Int {
        get { state.withLock(\.level) }
        set {
            assert(newValue >= 0, "Log level must be greater than or equal to 0")
            state.withLock { $0.level = newValue }
        }
    }

    /// Atomically replaces the active level and optional destination.
    ///
    /// A nil sink preserves the command-line tools' existing console behavior.
    public static func configure(level: Int, sink: Sink?) {
        precondition(level >= 0, "Log level must be greater than or equal to 0")
        state.withLock {
            $0.level = level
            $0.sink = sink
        }
    }

    /// Performs logging if enabled for the requested level.
    /// - Parameters:
    ///   - message: The message to log.
    ///   - component: The name of the component logging.
    ///   - level: The minimum log level to log at.
    public static func log(_ message: String, component: String? = nil, level: Int = 1) {
        let destination = state.withLock { current in
            (isEnabled: current.level >= level, sink: current.sink)
        }
        guard destination.isEnabled else {
            return
        }

        if let sink = destination.sink {
            sink(level, component, message)
            return
        }
        if let component {
            print("[\(component)] \(message)")
        } else {
            print(message)
        }
    }

    public static func isEnabled(at level: Int) -> Bool {
        state.withLock { $0.level >= level }
    }

    public static var isVerbose: Bool {
        Self.level >= 1
    }
}
