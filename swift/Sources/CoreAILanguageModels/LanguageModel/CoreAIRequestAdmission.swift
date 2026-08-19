// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Non-sensitive metrics for the exact request prepared by a Core AI executor.
///
/// The callback deliberately receives counts only. Prompt text, schemas, tool
/// arguments, attachment labels, and image bytes never cross this boundary.
public struct CoreAIRequestMetrics: Equatable, Sendable {
    public let inputTokenCount: Int
    public let reservedOutputTokenCount: Int
    public let attachmentCount: Int

    public init(
        inputTokenCount: Int,
        reservedOutputTokenCount: Int,
        attachmentCount: Int
    ) {
        self.inputTokenCount = inputTokenCount
        self.reservedOutputTokenCount = reservedOutputTokenCount
        self.attachmentCount = attachmentCount
    }
}

/// An async admission decision made after provider-native request preparation
/// and immediately before an inference engine starts generating.
///
/// Identity participates in executor configuration hashing so two otherwise
/// identical models with different admission policies never share an executor.
public struct CoreAIRequestAdmission: Hashable, Sendable {
    private let id: UUID
    private let body: @Sendable (CoreAIRequestMetrics) async throws -> Void

    public init(
        _ body: @Sendable @escaping (CoreAIRequestMetrics) async throws -> Void
    ) {
        self.id = UUID()
        self.body = body
    }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.id == rhs.id
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }

    func callAsFunction(_ metrics: CoreAIRequestMetrics) async throws {
        try Task.checkCancellation()
        try await body(metrics)
        try Task.checkCancellation()
    }

    static func perform<Result>(
        metrics: CoreAIRequestMetrics,
        admission: CoreAIRequestAdmission?,
        operation: () async throws -> Result
    ) async throws -> Result {
        if let admission {
            try await admission(metrics)
        } else {
            try Task.checkCancellation()
        }
        return try await operation()
    }
}
