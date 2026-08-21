// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// Foundation Models protocol implementation for VLM bundles.

import CoreAI
import CoreGraphics
import CoreImage
import Foundation
import FoundationModels
import ImageIO
import Tokenizers

// MARK: - CoreAIVisionLanguageModel

/// Foundation Models adapter for VLM bundles.
///
/// ```swift
/// let model = try await CoreAIVisionLanguageModel(
///     resourcesAt: vlmBundleURL, protocolProfile: .qwen35XML)
/// let session = LanguageModelSession(model: model)
/// let response = try await session.respond {
///     Prompt {
///         Attachment(image)
///         "What is in this image?"
///     }
/// }
/// ```
public struct CoreAIVisionLanguageModel: LanguageModel {
    public typealias Executor = CoreAIVLMExecutor

    /// The caller-supplied protocol this artifact speaks. Never inferred.
    let protocolProfile: CoreAILanguageProtocolProfile

    /// `.vision` is intrinsic to the adapter; everything else comes from the
    /// validated profile and nothing else.
    package static func declaredCapabilities(
        for profile: CoreAILanguageProtocolProfile
    ) -> [LanguageModelCapabilities.Capability] {
        [.vision] + profile.declaredCapabilities
    }

    public var capabilities: LanguageModelCapabilities {
        LanguageModelCapabilities(Self.declaredCapabilities(for: protocolProfile))
    }

    public var executorConfiguration: CoreAIVLMExecutor.Configuration

    /// Loads a VLM bundle and builds the backing engine.
    ///
    /// - Parameter url: URL to the bundle directory (`kind=vlm`).
    /// - Parameter protocolProfile: The transcript and generated-output
    ///   protocol this artifact speaks. Required, with no default. Validated
    ///   against the tokenizer — including its image convention — before any
    ///   engine is constructed.
    public init(
        resourcesAt url: URL,
        protocolProfile: CoreAILanguageProtocolProfile,
        requestAdmission: CoreAIRequestAdmission? = nil
    ) async throws {
        let bundle = try LanguageBundle(at: url)
        guard bundle.bundle.kind == .vlm else {
            throw InferenceRuntimeError.invalidArgument(
                "CoreAIVisionLanguageModel requires a VLM bundle (kind=vlm)")
        }
        guard let visionConfig = bundle.visionConfig else {
            throw InferenceRuntimeError.invalidArgument(
                "VLM bundle missing 'vision' config in metadata.json")
        }

        let visionURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.vision)
        let embedURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.embedding)
        let mainURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)

        let baseConfig = ModelConfig(
            name: bundle.name,
            tokenizer: bundle.tokenizer,
            vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength,
            serializedModel: [mainURL.path],
            function: bundle.language.functionMap?.name(for: "main") ?? "main"
        )
        let vlmConfig = VLMModelConfig(base: baseConfig, visionConfig: visionConfig)

        let tokenizer = try await bundle.loadTokenizer()

        // The profile gate. Both checks run before a single model component is
        // prepared, so a bundle whose template cannot serve the profile — or
        // has no image convention at all — fails without paying for a load.
        // Nothing below this line may move above it.
        let codec = CoreAITranscriptCodec(profile: protocolProfile)
        try codec.validate(tokenizer: tokenizer)
        try codec.validateVisionPairing(
            tokenizer: tokenizer, imageTokenID: visionConfig.imageTokenId)

        // Prepare the three model components concurrently.
        async let visionModelResult = PreparedModel.prepare(at: visionURL)
        async let embedModelResult = PreparedModel.prepare(at: embedURL)
        async let llmModelResult = PreparedModel.prepare(at: mainURL)

        let engine = try await CoreAISequentialVLMEngine(
            config: vlmConfig,
            visionModel: try await visionModelResult,
            embedModel: try await embedModelResult,
            llmModel: try await llmModelResult,
            options: EngineOptions()
        )

        self.protocolProfile = protocolProfile
        self.executorConfiguration = CoreAIVLMExecutor.Configuration(
            bundleURL: url,
            engine: engine,
            tokenizer: tokenizer,
            visionConfig: visionConfig,
            requestAdmission: requestAdmission
        )
    }
}

// MARK: - CoreAIVLMExecutor

public struct CoreAIVLMExecutor: LanguageModelExecutor {
    public typealias Model = CoreAIVisionLanguageModel

    public struct Configuration: Hashable, Sendable {
        let bundleURL: URL
        let engine: CoreAISequentialVLMEngine
        let tokenizer: any Tokenizer
        let visionConfig: VisionConfig
        let requestAdmission: CoreAIRequestAdmission?

        public static func == (lhs: Configuration, rhs: Configuration) -> Bool {
            lhs.bundleURL == rhs.bundleURL
                && lhs.requestAdmission == rhs.requestAdmission
        }
        public func hash(into hasher: inout Hasher) {
            hasher.combine(bundleURL)
            hasher.combine(requestAdmission)
        }
    }

    private let engine: CoreAISequentialVLMEngine
    private let tokenizer: any Tokenizer
    private let visionConfig: VisionConfig
    private let requestAdmission: CoreAIRequestAdmission?

    public init(configuration: Configuration) throws {
        self.engine = configuration.engine
        self.tokenizer = configuration.tokenizer
        self.visionConfig = configuration.visionConfig
        self.requestAdmission = configuration.requestAdmission
    }

    public nonisolated(nonsending) func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: CoreAIVisionLanguageModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        var images: [Transcript.ImageAttachment] = []
        var userText = ""
        for entry in request.transcript {
            guard case .prompt(let prompt) = entry else { continue }
            for segment in prompt.segments {
                switch segment {
                case .text(let text):
                    userText += text.content
                case .attachment(let attachment):
                    if case .image(let image) = attachment.content {
                        images.append(image)
                    }
                default:
                    break
                }
            }
        }

        try Self.validateImageCount(images.count)
        let cgImage = try Self.uprightCGImage(from: images[0])

        // Resolve the reasoning intent before generating: a profile with no
        // suppression mechanism refuses a disable request rather than quietly
        // generating a thought block anyway.
        let codec = CoreAITranscriptCodec(profile: model.protocolProfile)
        let reasoning = try codec.reasoningConfiguration(for: request.contextOptions.reasoningLevel)

        try await engine.reset()
        let embeddedInput = try await engine.encodeImage(cgImage: cgImage)

        let promptTokens = try Self.buildPromptTokens(
            userText: userText,
            imageTokenCount: embeddedInput.tokenCount,
            imageTokenId: visionConfig.imageTokenId,
            tokenizer: tokenizer,
            profile: model.protocolProfile,
            additionalContext: reasoning.additionalContext
        )

        let maxTokens = request.generationOptions.maximumResponseTokens ?? 512
        let metrics = CoreAIRequestMetrics(
            inputTokenCount: promptTokens.count,
            reservedOutputTokenCount: maxTokens,
            attachmentCount: images.count)
        var stopTokens = Set<Int32>()
        if let eos = tokenizer.eosTokenId { stopTokens.insert(Int32(eos)) }
        if let imEnd = tokenizer.convertTokenToId("<|im_end|>") { stopTokens.insert(Int32(imEnd)) }

        let stream = try await CoreAIRequestAdmission.perform(
            metrics: metrics,
            admission: requestAdmission
        ) {
            try await engine.generate(
                with: embeddedInput,
                tokens: promptTokens,
                samplingConfiguration: SamplingConfiguration(temperature: 1.0, topK: 1),
                inferenceOptions: InferenceOptions(maxTokens: maxTokens, includeLogits: false)
            )
        }

        // Same decoder the text adapter uses, driven by the same validated
        // profile, so a VLM that advertises reasoning or tool calling actually
        // segments them out instead of leaking protocol markup as response
        // text.
        var decoder = CoreAIStreamingOutputDecoder(
            profile: model.protocolProfile,
            reasoningEnabled: reasoning.enabled)
        var generatedCount = 0
        var reasoningTokenCount = 0
        var pendingTokens: [Int] = []
        var previousText = ""
        for try await output in stream {
            if stopTokens.contains(output.tokenId) { break }
            generatedCount += 1
            pendingTokens.append(Int(output.tokenId))

            let decoded = tokenizer.decode(tokens: pendingTokens)
            if decoded.unicodeScalars.contains("\u{FFFD}") {
                previousText = decoded
                continue
            }
            let common = decoded.commonPrefix(with: previousText)
            let delta = String(decoded.dropFirst(common.count))
            for event in try decoder.consume(delta) {
                if case .reasoning = event { reasoningTokenCount += 1 }
                await CoreAILanguageModel.CoreAIExecutor.dispatch(event, to: channel)
            }
            if let last = pendingTokens.last {
                pendingTokens = [last]
                previousText = tokenizer.decode(tokens: pendingTokens)
            }
        }

        // Drain anything held back waiting for a marker, and report an
        // unterminated block rather than dropping it.
        for event in try decoder.finish() {
            if case .reasoning = event { reasoningTokenCount += 1 }
            await CoreAILanguageModel.CoreAIExecutor.dispatch(event, to: channel)
        }

        await channel.send(
            .response(
                action: .updateUsage(
                    input: .init(totalTokenCount: promptTokens.count, cachedTokenCount: 0),
                    output: .init(
                        totalTokenCount: generatedCount,
                        reasoningTokenCount: reasoningTokenCount)
                )))
    }

    // MARK: - Prompt Construction

    static func uprightCGImage(
        from image: Transcript.ImageAttachment
    ) throws -> CGImage {
        let oriented = image.ciImage.oriented(image.orientation)
        let extent = oriented.extent.integral
        guard !extent.isEmpty,
            let rendered = CIContext(options: nil).createCGImage(oriented, from: extent)
        else {
            throw CoreAIVisionRequestError.imageRenderFailed
        }
        return rendered
    }

    static func validateImageCount(_ count: Int) throws {
        guard count == 1 else {
            throw CoreAIVisionRequestError.requiresExactlyOneImage(actualCount: count)
        }
    }

    /// Builds the token sequence for a single-image prompt.
    ///
    /// The composition — image token, newline, user text, as one user message
    /// put through the tokenizer's own chat template — is the same shape
    /// `CoreAITranscriptCodec.validateVisionPairing` proved at load, so what
    /// was validated is what runs. There is no hand-rolled prompt fallback: a
    /// tokenizer that cannot render this is rejected, since a guessed
    /// model-family format cannot be validated and silently produces a prompt
    /// the artifact never saw in training.
    static func buildPromptTokens(
        userText: String,
        imageTokenCount: Int,
        imageTokenId: Int32,
        tokenizer: any Tokenizer,
        profile: CoreAILanguageProtocolProfile,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int32] {
        let imageToken = tokenizer.convertIdToToken(Int(imageTokenId)) ?? "<|image_pad|>"
        let messages: [Message] = [
            ["role": "user", "content": "\(imageToken)\n\(userText)"]
        ]
        let templated: [Int]
        do {
            templated = try tokenizer.applyChatTemplate(
                messages: messages, tools: nil, additionalContext: additionalContext)
        } catch {
            throw CoreAIProtocolError(profile: profile, failure: .missingChatTemplate)
        }
        return try CoreAITranscriptCodec(profile: profile).expandImagePlaceholder(
            in: templated,
            imageTokenID: imageTokenId,
            imageTokenCount: imageTokenCount)
    }
}

/// Fail-closed request validation errors for the currently pinned Core AI VLM
/// adapter. Its backend supports one image per generation and does not accept a
/// text-only request.
public enum CoreAIVisionRequestError: Error, Equatable, Sendable {
    case requiresExactlyOneImage(actualCount: Int)
    case imageRenderFailed
}
