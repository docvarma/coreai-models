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

@Suite("Capability honesty")
struct CapabilityHonestyTests {
    @Test("Advertised reasoning always has evidence behind it")
    func reasoningAlwaysHasEvidence() {
        for profile in CoreAILanguageProtocolProfile.allCases {
            let codec = CoreAITranscriptCodec(profile: profile)
            if profile.supportsReasoning {
                #expect(!codec.reasoningEvidence.isEmpty)
            } else {
                #expect(codec.reasoningEvidence.isEmpty)
            }
        }
    }

    @Test("Advertised tool calling always has evidence behind it")
    func toolsAlwaysHaveEvidence() {
        for profile in CoreAILanguageProtocolProfile.allCases {
            let codec = CoreAITranscriptCodec(profile: profile)
            #expect(codec.toolEvidence.isEmpty == !profile.supportsToolCalling)
        }
    }

    @Test("Errors carry no content")
    func errorsCarryNoContent() {
        let error = CoreAIProtocolError(profile: .qwen35XML, failure: .malformedToolCall)
        let described = error.errorDescription ?? ""
        #expect(described.contains("qwen35XML"))
        #expect(described.contains("malformedToolCall"))
        // Nothing else may appear: no prompt, generated, tool, or image text.
        #expect(described.count < 200)
    }

    // MARK: - Capabilities come from the profile, not from the tokenizer

    /// Carries every marker the deleted `detectThinkingMarkers` and
    /// `detectToolCallMarkers` probes searched for. Under the guessing path a
    /// model built on this tokenizer advertised reasoning *and* tool calling
    /// no matter what it actually spoke, so any regression to probing turns
    /// `plainChat` into a reasoning, tool-calling model and fails this test.
    private static var markerRichTokenizer: MockTokenizer {
        MockTokenizer(vocab: [
            "<think>": 10, "</think>": 11,
            "<tool_call>": 12, "</tool_call>": 13,
        ])
    }

    private static func syntheticBundle() throws -> LanguageBundle {
        let dir = FileManager.default.temporaryDirectory.appending(
            path: "CapabilityHonestyTests-\(UUID().uuidString)/model")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try """
            {
              "metadata_version": "0.2",
              "kind": "llm",
              "name": "synthetic-bundle",
              "assets": { "main": "missing.aimodel" },
              "language": {
                "tokenizer": "synthetic/tokenizer",
                "vocab_size": 128,
                "max_context_length": 512
              }
            }
            """.write(to: dir.appending(path: "metadata.json"), atomically: true, encoding: .utf8)
        return try LanguageBundle(at: dir)
    }

    @Test("Capabilities follow the profile, not tokenizer markers")
    func capabilitiesFollowTheProfile() throws {
        let bundle = try Self.syntheticBundle()
        let configuration = CoreAILanguageModel.CoreAIExecutor.Configuration(
            url: bundle.bundlePath,
            variant: nil,
            kvCacheStrategy: .auto,
            modelIdentifier: bundle.name,
            samplingConfig: .greedy,
            vocabSize: bundle.vocabSize,
            requestAdmission: nil)
        for profile in CoreAILanguageProtocolProfile.allCases {
            let model = CoreAILanguageModel(
                configuration: configuration,
                bundle: bundle,
                tokenizer: Self.markerRichTokenizer,
                protocolProfile: profile,
                resources: ModelResources.shared(for: configuration),
                requestAdmission: nil)
            #expect(
                model.capabilities.contains(.reasoning) == profile.supportsReasoning,
                "\(profile.rawValue) advertised the wrong reasoning capability")
            #expect(
                model.capabilities.contains(.toolCalling) == profile.supportsToolCalling,
                "\(profile.rawValue) advertised the wrong tool-calling capability")
        }
    }

    @Test("The vision model keeps .vision and adds only what it can actually honor")
    func visionCapabilitiesFollowTheProfile() {
        for profile in CoreAILanguageProtocolProfile.allCases {
            let capabilities = LanguageModelCapabilities(
                CoreAIVisionLanguageModel.declaredCapabilities(for: profile))
            #expect(capabilities.contains(.vision), "\(profile.rawValue) dropped .vision")
            #expect(capabilities.contains(.reasoning) == profile.supportsReasoning)
            // Four of the five profiles have supportsToolCalling == true, so
            // this is a live assertion, not a tautology. The vision executor
            // never reads `request.enabledToolDefinitions` and never renders
            // them into the prompt, so advertising `.toolCalling` would accept
            // a session's tools and silently discard them.
            #expect(
                !capabilities.contains(.toolCalling),
                "\(profile.rawValue) advertised tools the vision executor cannot honor")
        }
    }

    // MARK: - The gate runs before the engine

    /// The bundle names a main asset that does not exist, so loading its
    /// engine cannot succeed. Its embedded tokenizer has no chat template at
    /// all, so the profile gate cannot succeed either. Only one of the two
    /// failures can be observed, and which one it is proves the ordering: a
    /// `CoreAIProtocolError` means validation ran first. Move the gate after
    /// the engine load — or delete it — and the engine's own load failure
    /// surfaces instead and this test fails.
    @Test("Profile validation rejects the pairing before any engine loads")
    func validationPrecedesEngineLoad() async throws {
        let tokenizerSource = try #require(
            TestResources.url(for: "MinimalTokenizer"),
            "MinimalTokenizer resource directory not found")
        let dir = FileManager.default.temporaryDirectory.appending(
            path: "CapabilityHonestyTests-\(UUID().uuidString)/model")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try """
            {
              "metadata_version": "0.2",
              "kind": "llm",
              "name": "synthetic-eager-bundle",
              "assets": { "main": "missing.aimodel" },
              "language": {
                "tokenizer": "synthetic/tokenizer",
                "vocab_size": 128,
                "max_context_length": 512
              }
            }
            """.write(to: dir.appending(path: "metadata.json"), atomically: true, encoding: .utf8)
        try FileManager.default.copyItem(at: tokenizerSource, to: dir.appending(path: "tokenizer"))

        await #expect(
            throws: CoreAIProtocolError(profile: .plainChat, failure: .missingChatTemplate)
        ) {
            _ = try await CoreAILanguageModel(
                resourcesAt: dir,
                protocolProfile: .plainChat,
                mode: .eager)
        }
    }
}

#endif
