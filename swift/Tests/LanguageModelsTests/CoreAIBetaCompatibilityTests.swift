// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Testing

@testable import CoreAILanguageModels

@Suite("Core AI beta compatibility")
struct CoreAIBetaCompatibilityTests {
    @Test("KV cache capacity derives a new shape without mutating the template")
    func kvCacheCapacityDerivesNewShape() {
        let template = [24, 1, 8, -1, 128]

        let resolved = KVCacheFactory.shape(
            template,
            replacingDimension: 3,
            with: 512
        )

        #expect(template == [24, 1, 8, -1, 128])
        #expect(resolved == [24, 1, 8, 512, 128])
    }

    @Test("Causal mask writer consumes its mutable view and fills the expected region")
    func causalMaskWriterFillsExpectedRegion() {
        var mask = NDArray(
            shape: [1, 4, 1, 2],
            scalarType: .float16
        )
        let maskView = mask.mutableView(as: LogitsScalarType.self)

        StaticShapeEngine.fillCausalMask(
            consume maskView,
            tokensInBatch: 2,
            alignedStep: 1
        )

        let values = mask.view(as: LogitsScalarType.self).withUnsafePointer { pointer, shape, strides in
            (0..<shape[1]).flatMap { context in
                (0..<shape[3]).map { query in
                    pointer[context * strides[1] + query * strides[3]]
                }
            }
        }

        #expect(values == [0, 0, 0, 0, -40000, 0, -40000, -40000])
    }
}
