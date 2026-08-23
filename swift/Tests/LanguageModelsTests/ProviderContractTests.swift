// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import FoundationModels
import TestUtilities
import Testing

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

@Suite("FoundationModels provider contract")
struct ProviderContractTests {
    private static func syntheticBundle() throws -> LanguageBundle {
        let directory = FileManager.default.temporaryDirectory.appending(
            path: "ProviderContractTests-\(UUID().uuidString)/model"
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try """
            {
              "metadata_version": "0.2",
              "kind": "llm",
              "name": "synthetic-provider-contract",
              "assets": { "main": "missing.aimodel" },
              "language": {
                "tokenizer": "synthetic/tokenizer",
                "vocab_size": 256,
                "max_context_length": 512
              }
            }
            """.write(
                to: directory.appending(path: "metadata.json"),
                atomically: true,
                encoding: .utf8
            )
        return try LanguageBundle(at: directory)
    }

    private static func model(engine: MockEngine) throws -> CoreAILanguageModel {
        let bundle = try syntheticBundle()
        let configuration = CoreAILanguageModel.CoreAIExecutor.Configuration(
            url: bundle.bundlePath,
            variant: nil,
            kvCacheStrategy: .auto,
            modelIdentifier: bundle.name,
            samplingConfig: .greedy,
            vocabSize: bundle.vocabSize,
            requestAdmission: nil
        )
        return CoreAILanguageModel(
            configuration: configuration,
            bundle: bundle,
            tokenizer: MockTokenizer(),
            protocolProfile: .plainChat,
            resources: ModelResources { engine },
            requestAdmission: nil
        )
    }

    @Test("Enabled tools take precedence over a response schema")
    func enabledToolsPrecedeSchema() {
        #expect(
            CoreAIRequestRouting.select(hasSchema: true, enabledToolCount: 1)
                == .protocolGeneration
        )
        #expect(
            CoreAIRequestRouting.select(hasSchema: true, enabledToolCount: 0)
                == .constrainedGeneration
        )
        #expect(
            CoreAIRequestRouting.select(hasSchema: false, enabledToolCount: 1)
                == .protocolGeneration
        )
    }

    @Test("Only EOS is a complete terminal reason")
    func terminalReasonsAreAuthoritative() {
        #expect(!CoreAILanguageModel.CoreAIExecutor.isTruncated(.eos))
        #expect(CoreAILanguageModel.CoreAIExecutor.isTruncated(.maxTokens))
        #expect(CoreAILanguageModel.CoreAIExecutor.isTruncated(.cancelled))
        #expect(CoreAILanguageModel.CoreAIExecutor.isTruncated(.error))
        #expect(CoreAILanguageModel.CoreAIExecutor.isTruncated(.stopSequence("synthetic")))
        #expect(CoreAILanguageModel.CoreAIExecutor.isTruncated(nil))
    }

    @Test("Maximum-token termination reaches LanguageModelSession usage")
    func maximumTokenTerminationReachesSessionUsage() async throws {
        let session = LanguageModelSession(
            model: try Self.model(engine: MockEngine(tokens: [65], vocabSize: 256))
        )
        var incompleteOutput = false
        for try await snapshot in session.streamResponse(
            to: "synthetic",
            options: GenerationOptions(maximumResponseTokens: 1)
        ) {
            if let value = snapshot.usage.metadata["incompleteOutput"] {
                incompleteOutput = (try? value.value(Bool.self)) == true
            }
        }
        #expect(incompleteOutput)
    }

    @Test("Natural EOS at the exact cap stays complete")
    func naturalEOSAtExactCapStaysComplete() async throws {
        let session = LanguageModelSession(
            model: try Self.model(engine: MockEngine(tokens: [65, 2], vocabSize: 256))
        )
        var incompleteOutput = false
        var content = ""
        for try await snapshot in session.streamResponse(
            to: "synthetic",
            options: GenerationOptions(maximumResponseTokens: 2)
        ) {
            content = snapshot.content
            if let value = snapshot.usage.metadata["incompleteOutput"] {
                incompleteOutput = (try? value.value(Bool.self)) == true
            }
        }
        #expect(content == "A")
        #expect(!incompleteOutput)
    }

    @Test("Sequential constrained decoding yields the grammar-closing token")
    func sequentialYieldsGrammarClosingToken() async throws {
        let strategy = ConstrainedDecodingStrategy(
            jsonSchema: #"{"type":"object","properties":{},"additionalProperties":false}"#,
            vocabSize: 256
        )
        let stream = try await strategy.decode(
            from: .tokens([1]),
            tokenizer: MockTokenizer(),
            inferenceEngine: MockEngine(tokens: [123, 125], vocabSize: 256),
            samplingConfiguration: .greedy,
            options: InferenceOptions(maxTokens: 8),
            stopSequences: StopSequences(sequences: [[2]])
        )

        var text = ""
        for try await result in stream {
            text += result.text
        }

        #expect(text == "{}")
    }
}

#endif
