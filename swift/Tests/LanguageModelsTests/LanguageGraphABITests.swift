// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Testing

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

@Suite("Language graph ABI")
struct LanguageGraphABITests {

    // MARK: - Acceptance

    @Test("Accepts a KV pair and up to two fixed persistent states", arguments: 0...2)
    func acceptsFixedPersistentStates(count: Int) throws {
        let validated = try LanguageGraphABI.validate(
            layout: makeLayout(extraStateCount: count),
            expectedVocabSize: 32
        )

        #expect(validated.keyCache.name == "state_0")
        #expect(validated.valueCache.name == "state_1")
        #expect(validated.persistentStates.count == count)
        #expect(validated.stateKinds["state_0"] == .kvCache)
        #expect(validated.stateKinds["state_1"] == .kvCache)
        for index in 0..<count {
            #expect(validated.stateKinds["state_\(index + 2)"] == .fixed)
        }
        #expect(validated.stateKinds.count == count + 2)
    }

    /// `StateKind.slidingCache` is deliberately unreachable from a validated
    /// ABI: a sliding window cache and a conv or recurrent state are both
    /// fixed-shape, and the only thing that told them apart was a
    /// model-family name guess. Every extra state is therefore `.fixed`,
    /// which sets `hasNonTruncatableStates` and forces full reset rather than
    /// prefix reuse. This pins that so restoring the capability means
    /// deliberately changing a failing test, not discovering the case is dead.
    @Test("A validated ABI never classifies a state as a sliding cache", arguments: 0...2)
    func slidingCacheIsUnreachable(count: Int) throws {
        let validated = try LanguageGraphABI.validate(
            layout: makeLayout(extraStateCount: count),
            expectedVocabSize: 32
        )
        #expect(!validated.stateKinds.values.contains(.slidingCache))
    }

    // MARK: - KV identification is shape-based

    // The KV pair used to be `states[0]` and `states[1]` on trust. Every
    // current exporter happens to emit `(key_cache, value_cache)` first, so
    // the ordering was an exporter accident rather than a checked property —
    // and `validateKVCache` never required a dynamic dimension, so a
    // fixed-shape state in position 0 was accepted as a "KV cache" while the
    // real cache was demoted.

    @Test("Rejects a fixed-shape state in the first KV position")
    func rejectsFixedStateInKVPosition() {
        // A rank-3 fixed conv state first, the real caches behind it. Before
        // the shape check this validated, bound the conv state as the KV
        // cache, and produced garbage with no error.
        let layout = LanguageGraphLayout(
            inputs: tokenInputs,
            outputs: logitsOutputs,
            states: [
                tensor("state_0", .float16, [1, 64, 8]),
                tensor("state_1", .float16, [1, 4, -1, 8]),
                tensor("state_2", .float16, [1, 4, -1, 8]),
            ]
        )
        expectRejection(layout, .output, containing: "first two states")
    }

    @Test("Rejects a graph whose states are all fixed-shape")
    func rejectsAllFixedStates() {
        // Nothing distinguishes a cache from a conv state here, so there is no
        // honest way to pick the pair. This is the shape the old positional
        // code silently mis-bound.
        let layout = LanguageGraphLayout(
            inputs: tokenInputs,
            outputs: logitsOutputs,
            states: [
                tensor("state_0", .float16, [1, 64, 8]),
                tensor("state_1", .float16, [1, 64, 8]),
                tensor("state_2", .float16, [1, 4, 16, 8]),
                tensor("state_3", .float16, [1, 4, 16, 8]),
            ]
        )
        expectRejection(layout, .output, containing: "first two states")
    }

    @Test("Rejects a dynamic persistent state")
    func rejectsDynamicPersistentState() {
        let layout = LanguageGraphLayout(
            inputs: tokenInputs,
            outputs: logitsOutputs,
            states: kvStates + [tensor("state_2", .float16, [1, -1, 64])]
        )
        expectRejection(layout, .output, containing: "first two states")
    }

    // MARK: - Persistent states

    @Test("Rejects more than two persistent states")
    func rejectsTooManyPersistentStates() {
        // Two KV states plus three extras. Pinning the message keeps this
        // honest: with only `#expect(throws: (any Error).self)` it passed on
        // whichever guard happened to fire first.
        expectRejection(
            makeLayout(extraStateCount: 3), .output,
            containing: "at most two persistent states")
    }

    @Test("Rejects a zero-dimensioned persistent state")
    func rejectsZeroDimensionedPersistentState() {
        let layout = LanguageGraphLayout(
            inputs: tokenInputs,
            outputs: logitsOutputs,
            states: kvStates + [tensor("state_2", .float16, [1, 0, 64])]
        )
        expectRejection(layout, .output, containing: "fixed, positive dimensions")
    }

    @Test("Rejects an unsupported persistent state scalar type")
    func rejectsUnsupportedPersistentStateScalar() {
        let layout = LanguageGraphLayout(
            inputs: tokenInputs,
            outputs: logitsOutputs,
            states: kvStates + [tensor("state_2", .int32, [1, 64, 8])]
        )
        expectRejection(layout, .output, containing: "persistent state scalar type is unsupported")
    }

    // MARK: - KV pair compatibility

    @Test("Rejects incompatible KV dynamic dimensions")
    func rejectsIncompatibleKVStates() {
        let layout = LanguageGraphLayout(
            inputs: tokenInputs,
            outputs: logitsOutputs,
            states: [
                tensor("state_0", .float16, [1, 4, -1, 8]),
                tensor("state_1", .float16, [1, -1, 4, 8]),
            ]
        )
        expectRejection(layout, .output, containing: "compatible scalar types and dynamic dimensions")
    }

    @Test("Rejects a rank-two KV state")
    func rejectsLowRankKVState() {
        let layout = LanguageGraphLayout(
            inputs: tokenInputs,
            outputs: logitsOutputs,
            states: [
                tensor("state_0", .float16, [1, -1]),
                tensor("state_1", .float16, [1, 4, -1, 8]),
            ]
        )
        expectRejection(layout, .output, containing: "rank-three-or-higher")
    }

    // MARK: - Inputs, outputs, names

    @Test("Rejects a non-int32 token input")
    func rejectsNonTokenInput() {
        let layout = LanguageGraphLayout(
            inputs: [tensor("input_0", .float16, [1, -1]), tokenInputs[1]],
            outputs: logitsOutputs,
            states: kvStates
        )
        expectRejection(layout, .input, containing: "token IDs must use int32")
    }

    @Test("Rejects a logits output that does not match the configured vocabulary")
    func rejectsMismatchedVocabulary() {
        expectRejection(
            makeLayout(), expectedVocabSize: 31, .output,
            containing: "configured vocabulary")
    }

    @Test("Rejects non-float16 logits")
    func rejectsNonFloat16Logits() {
        let layout = LanguageGraphLayout(
            inputs: tokenInputs,
            outputs: [tensor("output_0", .float32, [1, -1, 32])],
            states: kvStates
        )
        expectRejection(layout, .logits, containing: "logits must use float16")
    }

    @Test("Rejects duplicate graph tensor names")
    func rejectsDuplicateNames() {
        let layout = LanguageGraphLayout(
            inputs: tokenInputs,
            outputs: logitsOutputs,
            states: [
                tensor("input_0", .float16, [1, 4, -1, 8]),
                tensor("state_1", .float16, [1, 4, -1, 8]),
            ]
        )
        expectRejection(layout, .output, containing: "names must be unique")
    }

    @Test("Rejects a non-positive configured vocabulary size")
    func rejectsNonPositiveVocabulary() {
        expectRejection(
            makeLayout(), expectedVocabSize: 0, .output,
            containing: "positive configured vocabulary size")
    }

    // MARK: - The descriptor entry point

    // Every case above drives the `layout:` overload. Production calls the
    // `descriptor:` one, which was never executed by a test — including its
    // rejection of a tensor that is not an NDArray.

    @Test("The descriptor entry point validates the same graph as the layout one")
    func descriptorPathValidates() throws {
        let layout = makeLayout(extraStateCount: 1)
        let validated = try LanguageGraphABI.validate(
            descriptor: StubGraphDescriptor(layout: layout),
            expectedVocabSize: 32
        )
        #expect(validated.keyCache.name == "state_0")
        #expect(validated.valueCache.name == "state_1")
        #expect(validated.persistentStates.map(\.name) == ["state_2"])
        #expect(validated.inputIDs.name == "input_0")
        #expect(validated.positionIDs.name == "input_1")
        #expect(validated.logits.name == "output_0")
    }

    @Test("The descriptor entry point rejects an input that is not an NDArray")
    func descriptorRejectsNonNDArrayInput() {
        var stub = StubGraphDescriptor(layout: makeLayout())
        stub.inputs[0].tensor = nil
        expectRejection(stub, .input, containing: "input is not an NDArray")
    }

    @Test("The descriptor entry point rejects an output that is not an NDArray")
    func descriptorRejectsNonNDArrayOutput() {
        var stub = StubGraphDescriptor(layout: makeLayout())
        stub.outputs[0].tensor = nil
        expectRejection(stub, .output, containing: "output is not an NDArray")
    }

    @Test("The descriptor entry point rejects a state that is not an NDArray")
    func descriptorRejectsNonNDArrayState() {
        var stub = StubGraphDescriptor(layout: makeLayout())
        stub.states[1].tensor = nil
        expectRejection(stub, .output, containing: "state is not an NDArray")
    }

    // MARK: - Fixtures

    private var tokenInputs: [LanguageGraphLayout.Tensor] {
        [
            tensor("input_0", .int32, [1, -1]),
            tensor("input_1", .int32, [1, -1]),
        ]
    }

    private var logitsOutputs: [LanguageGraphLayout.Tensor] {
        [tensor("output_0", .float16, [1, -1, 32])]
    }

    private var kvStates: [LanguageGraphLayout.Tensor] {
        [
            tensor("state_0", .float16, [1, 4, -1, 8]),
            tensor("state_1", .float16, [1, 4, -1, 8]),
        ]
    }

    private func makeLayout(extraStateCount: Int = 0) -> LanguageGraphLayout {
        let extras = (0..<extraStateCount).map { index in
            tensor("state_\(index + 2)", .float16, [1, 64, 8])
        }
        return LanguageGraphLayout(
            inputs: tokenInputs,
            outputs: logitsOutputs,
            states: kvStates + extras
        )
    }

    private func tensor(
        _ name: String,
        _ scalarType: NDArray.ScalarType,
        _ shape: [Int]
    ) -> LanguageGraphLayout.Tensor {
        LanguageGraphLayout.Tensor(name: name, scalarType: scalarType, shape: shape)
    }

    // MARK: - Rejection assertions

    /// Which `InferenceRuntimeError` case a rejection used.
    private enum RejectionKind: Equatable {
        case input
        case output
        case logits
        case other
    }

    /// Asserts *which* guard rejected a fixture, not merely that something
    /// threw. `#expect(throws: (any Error).self)` cannot tell the guard a
    /// fixture was written for from an unrelated one it happens to trip, which
    /// is how a five-state fixture came to stand in for the persistent-state
    /// rules while asserting nothing about them.
    private func expectRejection(
        _ layout: LanguageGraphLayout,
        expectedVocabSize: Int = 32,
        _ kind: RejectionKind,
        containing fragment: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        expect(
            kind, fragment, sourceLocation: sourceLocation,
            performing: {
                try LanguageGraphABI.validate(
                    layout: layout, expectedVocabSize: expectedVocabSize)
            })
    }

    private func expectRejection(
        _ descriptor: StubGraphDescriptor,
        expectedVocabSize: Int = 32,
        _ kind: RejectionKind,
        containing fragment: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        expect(
            kind, fragment, sourceLocation: sourceLocation,
            performing: {
                try LanguageGraphABI.validate(
                    descriptor: descriptor, expectedVocabSize: expectedVocabSize)
            })
    }

    private func expect(
        _ kind: RejectionKind,
        _ fragment: String,
        sourceLocation: SourceLocation,
        performing body: () throws -> ValidatedLanguageGraphABI
    ) {
        do {
            _ = try body()
            Issue.record(
                "expected a \(kind) rejection mentioning \"\(fragment)\", but validation succeeded",
                sourceLocation: sourceLocation)
        } catch let error as InferenceRuntimeError {
            let (actualKind, message) = Self.describe(error)
            #expect(actualKind == kind, sourceLocation: sourceLocation)
            #expect(
                message.contains(fragment),
                "\"\(message)\" does not mention \"\(fragment)\"",
                sourceLocation: sourceLocation)
        } catch {
            Issue.record(
                "expected an InferenceRuntimeError, got \(type(of: error))",
                sourceLocation: sourceLocation)
        }
    }

    private static func describe(
        _ error: InferenceRuntimeError
    ) -> (RejectionKind, String) {
        switch error {
        case .invalidInputType(let message): return (.input, message)
        case .invalidOutputType(let message): return (.output, message)
        case .unsupportedLogitsType(let message): return (.logits, message)
        default: return (.other, error.errorDescription ?? "")
        }
    }
}

/// Stands in for `InferenceFunctionDescriptor`, which has no accessible
/// initializer. A `nil` tensor is a descriptor entry that is not an NDArray.
private struct StubGraphDescriptor: LanguageGraphFunctionDescriptor {
    struct Entry {
        var name: String
        var tensor: LanguageGraphLayout.Tensor?
    }

    var inputs: [Entry]
    var outputs: [Entry]
    var states: [Entry]

    init(layout: LanguageGraphLayout) {
        self.inputs = layout.inputs.map { Entry(name: $0.name, tensor: $0) }
        self.outputs = layout.outputs.map { Entry(name: $0.name, tensor: $0) }
        self.states = layout.states.map { Entry(name: $0.name, tensor: $0) }
    }

    var inputNames: [String] { inputs.map(\.name) }
    var outputNames: [String] { outputs.map(\.name) }
    var stateNames: [String] { states.map(\.name) }

    func ndArrayInput(named name: String) -> LanguageGraphLayout.Tensor? {
        inputs.first { $0.name == name }?.tensor
    }

    func ndArrayOutput(named name: String) -> LanguageGraphLayout.Tensor? {
        outputs.first { $0.name == name }?.tensor
    }

    func ndArrayState(named name: String) -> LanguageGraphLayout.Tensor? {
        states.first { $0.name == name }?.tensor
    }
}

#endif
