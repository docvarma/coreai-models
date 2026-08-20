// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared

public enum CoreAILogLevel: Int, Sendable {
    case off = 0
    case engine = 1
    case verbose = 2
    case trace = 3
}

public struct CoreAILogEvent: Sendable {
    public let level: CoreAILogLevel
    public let component: String?
    public let message: String

    public init(level: CoreAILogLevel, component: String?, message: String) {
        self.level = level
        self.component = component
        self.message = message
    }
}

public enum CoreAILogging {
    public static func configure(
        level: CoreAILogLevel,
        sink: (@Sendable (CoreAILogEvent) -> Void)?
    ) {
        guard let sink else {
            CLILogger.configure(level: level.rawValue, sink: nil)
            return
        }
        let sharedSink: CLILogger.Sink = { rawLevel, component, message in
            guard let eventLevel = CoreAILogLevel(rawValue: rawLevel) else {
                return
            }
            sink(CoreAILogEvent(level: eventLevel, component: component, message: message))
        }
        CLILogger.configure(level: level.rawValue, sink: sharedSink)
    }
}
