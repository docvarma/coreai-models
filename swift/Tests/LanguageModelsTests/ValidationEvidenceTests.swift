// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

@Suite("Validation evidence")
struct ValidationEvidenceTests {
    @Test("Every reasoning profile asserts reasoning evidence")
    func reasoningProfilesAssertReasoning() {
        for profile in CoreAILanguageProtocolProfile.allCases where profile.supportsReasoning {
            let codec = CoreAITranscriptCodec(profile: profile)
            #expect(
                !codec.reasoningEvidence.isEmpty,
                "\(profile.rawValue) advertises reasoning with no evidence")
        }
    }

    @Test("Every tool profile asserts tool evidence")
    func toolProfilesAssertTools() {
        for profile in CoreAILanguageProtocolProfile.allCases where profile.supportsToolCalling {
            let codec = CoreAITranscriptCodec(profile: profile)
            #expect(
                !codec.toolEvidence.isEmpty,
                "\(profile.rawValue) advertises tools with no evidence")
        }
    }

    @Test("plainChat asserts rendered fixture content")
    func plainChatAssertsContent() {
        let codec = CoreAITranscriptCodec(profile: .plainChat)
        let evidence = Set(codec.templateEvidence)
        #expect(evidence.isSuperset(of: ["synthetic-system", "synthetic-user", "synthetic-response"]))
    }

    @Test("plainChat rejects a template emitting reserved markers")
    func plainChatRejectsReservedMarkers() {
        let codec = CoreAITranscriptCodec(profile: .plainChat)
        for marker in CoreAITranscriptCodec.reservedMarkers {
            let rendered = "synthetic-system synthetic-user synthetic-response \(marker)"
            #expect(throws: CoreAIProtocolError.self) {
                try codec.assertEvidence(in: rendered)
            }
        }
    }

    @Test("Reserved markers cover every shipping profile's live markers")
    func reservedMarkersCoverShippingProfiles() {
        // Asserting concrete markers, not iterating the list itself: a test
        // that loops over `reservedMarkers` can never fail on an omission.
        let required = [
            "<think>", "</think>",              // qwen35XML reasoning
            "<tool_call>", "</tool_call>",      // qwen35XML tool block
            "<function=", "<parameter=",        // qwen35XML call body
            "<|channel|>",                      // harmony channels
            "<|channel>", "<channel|>",         // gemma4Channels thought
            "<|tool_call>", "<tool_call|>",     // gemma4Channels tool block
            "<|start|>", "<|message|>",         // harmony + atem envelopes
            "<|end|>", "<|return|>", "<|call|>",
            "<|eom|>", "<|eot|>",
            "<atem:function_calls>",
        ]
        let actual = Set(CoreAITranscriptCodec.reservedMarkers)
        for marker in required {
            #expect(actual.contains(marker), "reservedMarkers is missing \(marker)")
        }
    }

    @Test("plainChat accepts a clean template")
    func plainChatAcceptsCleanTemplate() throws {
        let codec = CoreAITranscriptCodec(profile: .plainChat)
        try codec.assertEvidence(in: "synthetic-system synthetic-user synthetic-response")
    }

    @Test("qwen35XML rejects a template that drops reasoning")
    func qwenRejectsDroppedReasoning() {
        let codec = CoreAITranscriptCodec(profile: .qwen35XML)
        // Tool evidence present, reasoning silently dropped by the template.
        let rendered = "<function=synthetic.tool><parameter=value>1 synthetic-result"
        #expect(throws: CoreAIProtocolError.self) {
            try codec.assertEvidence(in: rendered)
        }
    }
}

#endif
