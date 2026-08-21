// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Testing

@testable import CoreAILanguageModels

@Suite("Language graph ABI")
struct LanguageGraphABITests {
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
    }

    @Test("Rejects a dynamic persistent state")
    func rejectsDynamicPersistentState() {
        var layout = makeLayout(extraStateCount: 1)
        layout = LanguageGraphLayout(
            inputs: layout.inputs,
            outputs: layout.outputs,
            states: Array(layout.states.prefix(2)) + [
                tensor("state_2", .float16, [1, -1, 64])
            ]
        )

        #expect(throws: (any Error).self) {
            try LanguageGraphABI.validate(layout: layout, expectedVocabSize: 32)
        }
    }

    @Test("Rejects more than two persistent states")
    func rejectsTooManyPersistentStates() {
        #expect(throws: (any Error).self) {
            try LanguageGraphABI.validate(
                layout: makeLayout(extraStateCount: 3),
                expectedVocabSize: 32
            )
        }
    }

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

        #expect(throws: (any Error).self) {
            try LanguageGraphABI.validate(layout: layout, expectedVocabSize: 32)
        }
    }

    @Test("Rejects non-token inputs and a mismatched vocabulary")
    func rejectsInvalidIO() {
        let wrongInput = LanguageGraphLayout(
            inputs: [
                tensor("input_0", .float16, [1, -1]),
                tokenInputs[1],
            ],
            outputs: logitsOutputs,
            states: Array(makeLayout().states.prefix(2))
        )
        #expect(throws: (any Error).self) {
            try LanguageGraphABI.validate(layout: wrongInput, expectedVocabSize: 32)
        }

        #expect(throws: (any Error).self) {
            try LanguageGraphABI.validate(layout: makeLayout(), expectedVocabSize: 31)
        }
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

        #expect(throws: (any Error).self) {
            try LanguageGraphABI.validate(layout: layout, expectedVocabSize: 32)
        }
    }

    private var tokenInputs: [LanguageGraphLayout.Tensor] {
        [
            tensor("input_0", .int32, [1, -1]),
            tensor("input_1", .int32, [1, -1]),
        ]
    }

    private var logitsOutputs: [LanguageGraphLayout.Tensor] {
        [tensor("output_0", .float16, [1, -1, 32])]
    }

    private func makeLayout(extraStateCount: Int = 0) -> LanguageGraphLayout {
        let kvStates = [
            tensor("state_0", .float16, [1, 4, -1, 8]),
            tensor("state_1", .float16, [1, 4, -1, 8]),
        ]
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
}
