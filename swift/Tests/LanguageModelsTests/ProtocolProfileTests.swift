// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

@Suite("CoreAILanguageProtocolProfile")
struct ProtocolProfileTests {
    @Test("Only qualifiable profiles ship")
    func shippingSetIsExact() {
        let names = Set(CoreAILanguageProtocolProfile.allCases.map(\.rawValue))
        #expect(names == ["plainChat", "qwen35XML", "harmony", "gemma4Channels", "atem"])
    }

    @Test("Deferred profiles are absent")
    func deferredProfilesAbsent() {
        for name in ["qwen3Tags", "mistralToolArray", "llama3Chat", "llama3Tools"] {
            #expect(CoreAILanguageProtocolProfile(rawValue: name) == nil)
        }
    }

    @Test("Every reasoning profile declares a disabling policy deliberately")
    func reasoningPolicyIsExplicit() {
        #expect(CoreAILanguageProtocolProfile.qwen35XML.supportsDisablingReasoning)
        #expect(CoreAILanguageProtocolProfile.gemma4Channels.supportsDisablingReasoning)
        // Harmony and ATEM have no suppression mechanism in their protocol.
        #expect(!CoreAILanguageProtocolProfile.harmony.supportsDisablingReasoning)
        #expect(!CoreAILanguageProtocolProfile.atem.supportsDisablingReasoning)
    }

    @Test("plainChat advertises no structural capability")
    func plainChatIsInert() {
        #expect(!CoreAILanguageProtocolProfile.plainChat.supportsReasoning)
        #expect(!CoreAILanguageProtocolProfile.plainChat.supportsToolCalling)
    }
}

#endif
