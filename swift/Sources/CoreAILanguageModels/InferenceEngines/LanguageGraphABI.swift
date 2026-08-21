// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI

/// The closed tensor layout accepted by the text-generation engines.
///
/// Tensor names are opaque graph identity. Their ordered descriptor positions,
/// scalar types, and shapes define the ABI; model-family or token-name guesses
/// never participate in validation.
struct LanguageGraphLayout {
    struct Tensor {
        let name: String
        let scalarType: NDArray.ScalarType
        let shape: [Int]
    }

    let inputs: [Tensor]
    let outputs: [Tensor]
    let states: [Tensor]
}

struct ValidatedLanguageGraphABI {
    let inputIDs: LanguageGraphLayout.Tensor
    let positionIDs: LanguageGraphLayout.Tensor
    let logits: LanguageGraphLayout.Tensor
    let keyCache: LanguageGraphLayout.Tensor
    let valueCache: LanguageGraphLayout.Tensor
    let persistentStates: [LanguageGraphLayout.Tensor]

    /// How each validated state is handled downstream.
    ///
    /// Deliberately only `.kvCache` and `.fixed`. `StateKind.slidingCache`
    /// exists and `StateHandlerFactory` still honours it, but nothing here can
    /// produce it, and that is a decision rather than an oversight: a sliding
    /// window cache and a conv or recurrent state are both fixed-shape, so
    /// telling them apart needs a declaration, and the only declarations
    /// available were the model-family name guesses this provider exists to
    /// delete (`name.contains("cache")`). Until a bundle can *declare* a
    /// sliding cache, every non-KV state is classified `.fixed`, which sets
    /// `hasNonTruncatableStates` and forces full reset instead of prefix
    /// reuse — correct but pessimistic for a hybrid or sliding-window model.
    ///
    /// `slidingCacheIsUnreachable` in the ABI tests pins this, so restoring
    /// the capability is a deliberate change to a failing test rather than a
    /// silent one.
    var stateKinds: [String: StateKind] {
        var kinds = [keyCache.name: StateKind.kvCache, valueCache.name: .kvCache]
        for state in persistentStates {
            kinds[state.name] = .fixed
        }
        return kinds
    }
}

/// The slice of a compiled function's descriptor the language graph ABI reads.
///
/// `InferenceFunctionDescriptor` and `NDArrayDescriptor` have no accessible
/// initializers, so the production entry point below could not be driven by a
/// test at all while it took one directly. A tensor that is not an NDArray
/// surfaces here as `nil` and `validate` turns that into the typed rejection,
/// which leaves only the one-line enum destructuring in the conformance
/// outside test reach.
protocol LanguageGraphFunctionDescriptor {
    var inputNames: [String] { get }
    var outputNames: [String] { get }
    var stateNames: [String] { get }
    func ndArrayInput(named name: String) -> LanguageGraphLayout.Tensor?
    func ndArrayOutput(named name: String) -> LanguageGraphLayout.Tensor?
    func ndArrayState(named name: String) -> LanguageGraphLayout.Tensor?
}

extension InferenceFunctionDescriptor: LanguageGraphFunctionDescriptor {
    func ndArrayInput(named name: String) -> LanguageGraphLayout.Tensor? {
        Self.tensor(named: name, from: inputDescriptor(of: name))
    }

    func ndArrayOutput(named name: String) -> LanguageGraphLayout.Tensor? {
        Self.tensor(named: name, from: outputDescriptor(of: name))
    }

    func ndArrayState(named name: String) -> LanguageGraphLayout.Tensor? {
        Self.tensor(named: name, from: stateDescriptor(of: name))
    }

    private static func tensor(
        named name: String,
        from descriptor: InferenceValue.Descriptor?
    ) -> LanguageGraphLayout.Tensor? {
        guard case .ndArray(let value) = descriptor else { return nil }
        return LanguageGraphLayout.Tensor(
            name: name,
            scalarType: value.scalarType,
            shape: value.shape
        )
    }
}

enum LanguageGraphABI {
    static func validate(
        descriptor: some LanguageGraphFunctionDescriptor,
        expectedVocabSize: Int
    ) throws -> ValidatedLanguageGraphABI {
        let inputs = try descriptor.inputNames.map { name in
            guard let tensor = descriptor.ndArrayInput(named: name) else {
                throw InferenceRuntimeError.invalidInputType(
                    "Language graph input is not an NDArray"
                )
            }
            return tensor
        }
        let outputs = try descriptor.outputNames.map { name in
            guard let tensor = descriptor.ndArrayOutput(named: name) else {
                throw InferenceRuntimeError.invalidOutputType(
                    "Language graph output is not an NDArray"
                )
            }
            return tensor
        }
        let states = try descriptor.stateNames.map { name in
            guard let tensor = descriptor.ndArrayState(named: name) else {
                throw InferenceRuntimeError.invalidOutputType(
                    "Language graph state is not an NDArray"
                )
            }
            return tensor
        }
        let layout = LanguageGraphLayout(
            inputs: inputs,
            outputs: outputs,
            states: states
        )
        return try validate(layout: layout, expectedVocabSize: expectedVocabSize)
    }

    static func validate(
        layout: LanguageGraphLayout,
        expectedVocabSize: Int
    ) throws -> ValidatedLanguageGraphABI {
        guard layout.inputs.count == 2 else {
            throw InferenceRuntimeError.invalidInputType(
                "Language graph requires exactly two token inputs; found \(layout.inputs.count)"
            )
        }
        guard layout.outputs.count == 1 else {
            throw InferenceRuntimeError.invalidOutputType(
                "Language graph requires exactly one logits output; found \(layout.outputs.count)"
            )
        }
        guard (2...4).contains(layout.states.count) else {
            throw InferenceRuntimeError.invalidOutputType(
                "Language graph requires two KV states and at most two persistent states; "
                    + "found \(layout.states.count) states"
            )
        }
        guard expectedVocabSize > 0 else {
            throw InferenceRuntimeError.invalidOutputType(
                "Language graph requires a positive configured vocabulary size"
            )
        }

        let allNames = (layout.inputs + layout.outputs + layout.states).map(\.name)
        guard Set(allNames).count == allNames.count else {
            throw InferenceRuntimeError.invalidOutputType(
                "Language graph tensor names must be unique"
            )
        }

        let inputIDs = layout.inputs[0]
        let positionIDs = layout.inputs[1]
        try validateTokenInput(inputIDs, role: "token IDs")
        try validateTokenInput(positionIDs, role: "position IDs")

        let logits = layout.outputs[0]
        guard logits.scalarType == .float16 else {
            throw InferenceRuntimeError.unsupportedLogitsType(
                "Language graph logits must use float16"
            )
        }
        guard logits.shape.count == 3,
            logits.shape[0] == 1,
            isPositiveOrDynamic(logits.shape[1]),
            logits.shape[2] == expectedVocabSize
        else {
            throw InferenceRuntimeError.invalidOutputType(
                "Language graph logits must have shape [1, sequence, configured vocabulary]"
            )
        }

        // The KV pair is identified by shape, not by position. Taking
        // `states[0]` and `states[1]` on trust bound whatever the exporter
        // happened to emit first: a graph that lists a fixed conv state first
        // had that state bound as its KV cache and the real cache demoted to a
        // persistent state, with garbage output and no error anywhere. A
        // dynamic dimension is what makes a state a growing cache, and the
        // persistent states are required to be fixed just below, so requiring
        // the dynamically shaped states to be exactly the first two states
        // makes the ordering a checked property instead of an exporter
        // accident.
        let dynamicStateIndices = layout.states.indices.filter { index in
            !dynamicDimensions(in: layout.states[index].shape).isEmpty
        }
        guard dynamicStateIndices == [0, 1] else {
            throw InferenceRuntimeError.invalidOutputType(
                "Language graph requires its two dynamically shaped KV states to be the "
                    + "first two states"
            )
        }

        let keyCache = layout.states[0]
        let valueCache = layout.states[1]
        try validateKVCache(keyCache)
        try validateKVCache(valueCache)
        guard keyCache.scalarType == valueCache.scalarType,
            keyCache.shape.count == valueCache.shape.count,
            dynamicDimensions(in: keyCache.shape) == dynamicDimensions(in: valueCache.shape)
        else {
            throw InferenceRuntimeError.invalidOutputType(
                "Language graph KV states must use compatible scalar types and dynamic dimensions"
            )
        }

        let persistentStates = Array(layout.states.dropFirst(2))
        for state in persistentStates {
            guard state.shape.allSatisfy({ $0 > 0 }) else {
                throw InferenceRuntimeError.invalidOutputType(
                    "Language graph persistent states must have fixed, positive dimensions"
                )
            }
            guard isSupportedStateScalar(state.scalarType) else {
                throw InferenceRuntimeError.invalidOutputType(
                    "Language graph persistent state scalar type is unsupported"
                )
            }
        }

        return ValidatedLanguageGraphABI(
            inputIDs: inputIDs,
            positionIDs: positionIDs,
            logits: logits,
            keyCache: keyCache,
            valueCache: valueCache,
            persistentStates: persistentStates
        )
    }

    private static func validateTokenInput(
        _ tensor: LanguageGraphLayout.Tensor,
        role: String
    ) throws {
        guard tensor.scalarType == .int32 else {
            throw InferenceRuntimeError.invalidInputType(
                "Language graph \(role) must use int32"
            )
        }
        guard tensor.shape.count == 2,
            tensor.shape[0] == 1,
            isPositiveOrDynamic(tensor.shape[1])
        else {
            throw InferenceRuntimeError.invalidInputType(
                "Language graph \(role) must have shape [1, sequence]"
            )
        }
    }

    private static func validateKVCache(_ tensor: LanguageGraphLayout.Tensor) throws {
        guard tensor.shape.count >= 3,
            tensor.shape.allSatisfy(isPositiveOrDynamic),
            isSupportedStateScalar(tensor.scalarType)
        else {
            throw InferenceRuntimeError.invalidOutputType(
                "Language graph KV states require supported scalars and nonzero rank-three-or-higher shapes"
            )
        }
    }

    private static func dynamicDimensions(in shape: [Int]) -> [Int] {
        shape.indices.filter { shape[$0] < 0 }
    }

    private static func isPositiveOrDynamic(_ dimension: Int) -> Bool {
        dimension > 0 || dimension == -1
    }

    private static func isSupportedStateScalar(_ scalarType: NDArray.ScalarType) -> Bool {
        scalarType == .float16 || scalarType == .bfloat16 || scalarType == .float32
    }
}
