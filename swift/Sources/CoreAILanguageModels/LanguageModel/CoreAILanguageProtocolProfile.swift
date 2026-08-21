// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import FoundationModels

/// A closed description of the transcript and generated-output protocol used
/// by a precompiled Core AI language model.
///
/// The profile is explicit installation identity. It is never inferred from a
/// repository name, tokenizer vocabulary, or generated marker.
public enum CoreAILanguageProtocolProfile: String, CaseIterable, Codable, Sendable {
    case plainChat
    case qwen35XML
    case harmony
    case gemma4Channels
    case atem

    package var supportsReasoning: Bool {
        switch self {
        case .qwen35XML, .harmony, .gemma4Channels, .atem: true
        case .plainChat: false
        }
    }

    package var supportsToolCalling: Bool {
        switch self {
        case .qwen35XML, .harmony, .gemma4Channels, .atem: true
        case .plainChat: false
        }
    }

    package var defaultReasoningEnabled: Bool {
        switch self {
        case .qwen35XML, .harmony, .atem: true
        case .plainChat, .gemma4Channels: false
        }
    }

    /// False where the protocol has no suppression mechanism at all.
    package var supportsDisablingReasoning: Bool {
        switch self {
        case .qwen35XML, .gemma4Channels: true
        case .plainChat, .harmony, .atem: false
        }
    }

    /// The FoundationModels capabilities implied by the profile alone. This is
    /// the only source of `.reasoning` and `.toolCalling` for either adapter:
    /// nothing is inferred from the tokenizer's vocabulary.
    package var declaredCapabilities: [LanguageModelCapabilities.Capability] {
        var capabilities: [LanguageModelCapabilities.Capability] = []
        if supportsReasoning { capabilities.append(.reasoning) }
        if supportsToolCalling { capabilities.append(.toolCalling) }
        return capabilities
    }
}

/// Stable, PHI-safe reason codes for provider protocol failures.
public enum CoreAIProtocolFailure: String, Codable, Sendable {
    case missingChatTemplate
    case incompatibleChatTemplate
    case unsupportedTranscriptContent
    case invalidToolDefinition
    case invalidPriorToolCall
    case unfinishedReasoning
    case unfinishedToolCall
    case nestedProtocolBlock
    case duplicateProtocolBlock
    case mixedResponseAndToolCall
    case malformedReasoning
    case malformedToolCall
    case malformedChannel
    case unsupportedReasoningPolicy
    case missingImagePlaceholder
    case duplicateImagePlaceholder
}

/// A typed provider error that identifies only the selected profile and a
/// stable failure reason. It never includes prompt, generated, tool, or image
/// content.
public struct CoreAIProtocolError: Error, Equatable, Sendable, LocalizedError {
    public let profile: CoreAILanguageProtocolProfile
    public let failure: CoreAIProtocolFailure

    public init(
        profile: CoreAILanguageProtocolProfile,
        failure: CoreAIProtocolFailure
    ) {
        self.profile = profile
        self.failure = failure
    }

    public var errorDescription: String? {
        "Core AI protocol \(profile.rawValue) failed validation (\(failure.rawValue))."
    }
}
