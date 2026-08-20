// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILanguageModels
import CoreAIShared
import Synchronization
import Testing

@Suite("Core AI logging", .serialized)
struct CoreAILoggingTests {
    @Test("Levels filter events and preserve their metadata")
    func levelsFilterEvents() {
        let events = Mutex<[CoreAILogEvent]>([])
        defer { CoreAILogging.configure(level: .off, sink: nil) }

        for configuredLevel in CoreAILogLevel.allCasesForTesting {
            events.withLock { $0.removeAll() }
            CoreAILogging.configure(level: configuredLevel) { event in
                events.withLock { $0.append(event) }
            }

            for emittedLevel in 1 ... 3 {
                CLILogger.log(
                    "event-\(emittedLevel)",
                    component: "test",
                    level: emittedLevel
                )
            }

            let captured = events.withLock { $0 }
            let expectedLevels = Array(1 ... 3).filter { $0 <= configuredLevel.rawValue }
            #expect(captured.map(\.level.rawValue) == expectedLevels)
            #expect(captured.allSatisfy { $0.component == "test" })
            #expect(captured.map(\.message) == expectedLevels.map { "event-\($0)" })
        }
    }

    @Test("Concurrent delivery does not lose events")
    func concurrentDeliveryIsThreadSafe() async {
        let count = 500
        let received = Mutex<[String]>([])
        defer { CoreAILogging.configure(level: .off, sink: nil) }
        CoreAILogging.configure(level: .trace) { event in
            received.withLock { $0.append(event.message) }
        }

        await withTaskGroup(of: Void.self) { group in
            for index in 0 ..< count {
                group.addTask {
                    CLILogger.log("logging-test-\(index)", level: 2)
                }
            }
        }

        let messages = received.withLock { messages in
            messages.filter { $0.hasPrefix("logging-test-") }
        }
        #expect(messages.count == count)
        #expect(Set(messages).count == count)
    }
}

private extension CoreAILogLevel {
    static let allCasesForTesting: [Self] = [.off, .engine, .verbose, .trace]
}
