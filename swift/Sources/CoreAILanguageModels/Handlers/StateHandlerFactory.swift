// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared

/// Classification of a model state's lifecycle behavior.
public enum StateKind: String, Codable, Sendable {
    /// KV cache — grows dynamically with context, supports truncation (causal mask).
    case kvCache = "kv_cache"
    /// Sliding window cache — fixed size, supports truncation (causal mask).
    case slidingCache = "sliding_cache"
    /// Fixed state (conv, recurrent) — fixed size, does NOT support truncation.
    case fixed
}

/// Result of state handler creation.
struct SyncStateHandlerSet {
    /// Growing states (KV caches with dynamic sequence dimension).
    var kvCache: any SyncStateHandler
    /// Fixed states (sliding caches + recurrent/conv). Nil for transformer-only models.
    var additionalStates: FixedNDArrayState?
    /// Whether any state is non-truncatable (triggers full-reset-only mode).
    var hasNonTruncatableStates: Bool
}

/// Creates state handlers from an explicitly validated model descriptor.
enum StateHandlerFactory {
    /// Create sync state handlers from classified states.
    static func createSyncHandlers(
        descriptor: InferenceFunctionDescriptor,
        maxContextLength: Int,
        stateKinds: [String: StateKind],
        options: EngineOptions = EngineOptions()
    ) throws -> SyncStateHandlerSet {
        guard !descriptor.stateNames.isEmpty else {
            throw InferenceRuntimeError.invalidOutputType(
                "Expected states but found none")
        }

        guard Set(stateKinds.keys) == Set(descriptor.stateNames) else {
            throw InferenceRuntimeError.invalidOutputType(
                "Validated state roles must match the graph states exactly"
            )
        }
        let classified = descriptor.stateNames.map { name in
            (name: name, kind: stateKinds[name]!)
        }

        // Separate into growing (kvCache) and fixed (slidingCache + fixed)
        var growingPairs: [(name: String, descriptor: NDArrayDescriptor)] = []
        var fixedPairs: [(name: String, descriptor: NDArrayDescriptor)] = []
        var hasNonTruncatable = false

        for (name, kind) in classified {
            guard case .ndArray(let desc) = descriptor.stateDescriptor(of: name) else {
                throw InferenceRuntimeError.invalidOutputType(
                    "Cannot get state descriptor for '\(name)'")
            }

            switch kind {
            case .kvCache:
                growingPairs.append((name, desc))
            case .slidingCache:
                fixedPairs.append((name, desc))
            case .fixed:
                fixedPairs.append((name, desc))
                hasNonTruncatable = true
            }
        }

        // Build growing handler (KV caches)
        let kvCache: any SyncStateHandler
        if !growingPairs.isEmpty {
            let hasDynamicKVShape = growingPairs.contains { pair in
                pair.descriptor.shape.contains(where: { $0 < 0 })
            }
            if options.kvCacheStrategy == .fixedSize || !hasDynamicKVShape {
                let resolved = growingPairs.map { (name, desc) -> (name: String, descriptor: NDArrayDescriptor) in
                    let resolvedDesc = desc.resolvingDynamicDimensions(
                        desc.shape.map { $0 < 0 ? maxContextLength : $0 })
                    return (name, resolvedDesc)
                }
                kvCache = FixedNDArrayState(states: resolved)
            } else {
                let initial = min(256, maxContextLength)
                kvCache = GrowingNDArrayState(
                    states: growingPairs,
                    initialCapacity: initial,
                    maxCapacity: maxContextLength
                )
            }
        } else {
            // All states are fixed (e.g., all sliding caches at fixed-size mode)
            // Use the fixed pairs as KV cache too
            kvCache = FixedNDArrayState(states: fixedPairs)
            fixedPairs = []
        }

        // Build fixed handler (sliding caches + recurrent/conv)
        var additionalStates: FixedNDArrayState? = nil
        if !fixedPairs.isEmpty {
            additionalStates = FixedNDArrayState(states: fixedPairs)
        }

        return SyncStateHandlerSet(
            kvCache: kvCache,
            additionalStates: additionalStates,
            hasNonTruncatableStates: hasNonTruncatable
        )
    }
}
