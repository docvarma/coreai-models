// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import FoundationModels
import Synchronization
import Tokenizers

enum CoreAIRequestRouting: Equatable, Sendable {
    case protocolGeneration
    case constrainedGeneration

    static func select(hasSchema: Bool, enabledToolCount: Int) -> Self {
        if enabledToolCount > 0 || !hasSchema {
            return .protocolGeneration
        }
        return .constrainedGeneration
    }
}

/// FoundationModels Adoption for Core AI inference engines.
///
/// Wraps any `InferenceEngine` (pipelined, sequential, or static-shape) and exposes it
/// through the FoundationModels `LanguageModel` protocol. It uses the modern `tokenSequence()`
/// API for efficient streaming token generation.
/// ## Engine Selection
/// The engine type is determined by `EngineFactory` based on model structure:
/// - **Pipelined**: GPU-accelerated with pipeline-depth-matched buffering (fastest for GPU models)
/// - **Sequential**: CPU-based synchronous execution (fallback)
/// - **Static-shape**: Neural Engine optimized for chunked static models
///
/// ## Usage
/// ```swift
/// let model = try await CoreAILanguageModel(
///     resourcesAt: url, protocolProfile: .qwen35XML)  // .lazy by default
/// print(model.estimatedSizeOnDiskBytes ?? 0)
/// try await model.load()                                       // optional; respond auto-loads
/// let session = LanguageModelSession(model: model)
/// // ... generate ...
/// model.unload()
/// ```
public struct CoreAILanguageModel: LanguageModel {
    public enum LoadMode: Sendable {
        case lazy
        case eager
    }

    // MARK: - Properties

    private let url: URL
    private let variant: String?
    private let kvCacheStrategy: KVCacheStrategy
    fileprivate let samplingConfig: SamplingConfiguration
    fileprivate let bundle: LanguageBundle
    fileprivate let tokenizer: any Tokenizer
    /// The caller-supplied protocol this artifact speaks. Never inferred.
    fileprivate let protocolProfile: CoreAILanguageProtocolProfile
    /// Built from `protocolProfile` and validated against the tokenizer before
    /// the model exists; it owns every transcript rendering decision.
    fileprivate let codec: CoreAITranscriptCodec
    fileprivate let resources: ModelResources
    fileprivate let requestAdmission: CoreAIRequestAdmission?
    /// All EOS-like token IDs beyond the tokenizer's main `eosTokenId` — e.g.
    /// Gemma's `<end_of_turn>`, read from tokenizer_config.json at init.
    fileprivate let additionalEosTokenIds: [Int32]

    // MARK: - Protocol Requirements

    public typealias Executor = CoreAIExecutor

    /// Reasoning and tool calling come from the validated profile and nothing
    /// else; guided generation remains a property of the loaded engine.
    public var capabilities: LanguageModelCapabilities {
        var caps = protocolProfile.declaredCapabilities
        if isGuidedGenerationSupported { caps.append(.guidedGeneration) }
        return LanguageModelCapabilities(caps)
    }

    public var executorConfiguration: CoreAIExecutor.Configuration {
        CoreAIExecutor.Configuration(
            url: url,
            variant: variant,
            kvCacheStrategy: kvCacheStrategy,
            modelIdentifier: bundle.name,
            samplingConfig: samplingConfig,
            vocabSize: bundle.vocabSize,
            requestAdmission: requestAdmission
        )
    }

    // MARK: - Initialization

    /// Creates a model from a resource bundle on disk.
    ///
    /// ```swift
    /// let model = try await CoreAILanguageModel(
    ///     resourcesAt: url, protocolProfile: .qwen35XML)              // lazy
    /// let model = try await CoreAILanguageModel(
    ///     resourcesAt: url, protocolProfile: .qwen35XML, mode: .eager)
    /// ```
    ///
    /// - Parameter url: URL to the model bundle directory.
    /// - Parameter protocolProfile: The transcript and generated-output
    ///   protocol this artifact speaks. Required, with no default: the
    ///   provider never infers it. Validated against the tokenizer's chat
    ///   template before any engine is constructed.
    /// - Parameter mode: When to load the engine. Defaults to `.lazy`. With
    ///   `.eager`, the engine loads once the profile gate has passed.
    /// - Parameter variant: Engine variant override (e.g. "coreai-sequential",
    ///   "ane"). Nil for auto-detect from model structure.
    /// - Parameter kvCacheStrategy: KV cache memory strategy. Defaults to
    ///   `.auto` (256-token initial size for dynamic models). Pass
    ///   `.fixedSize` to pre-allocate at full `maxContextLength`.
    /// - Throws: `CoreAIProtocolError` when the tokenizer cannot honor the
    ///   selected profile, or if the asset bundle is invalid or the tokenizer
    ///   fails to load. With `.eager`, also throws on engine-creation failure.
    public init(
        resourcesAt url: URL,
        protocolProfile: CoreAILanguageProtocolProfile,
        mode: LoadMode = .lazy,
        variant: String? = nil,
        kvCacheStrategy: KVCacheStrategy = .auto,
        requestAdmission: CoreAIRequestAdmission? = nil
    ) async throws {
        let bundle = try LanguageBundle(at: url)
        let configuration = CoreAIExecutor.Configuration(
            url: url,
            variant: variant,
            kvCacheStrategy: kvCacheStrategy,
            modelIdentifier: bundle.name,
            samplingConfig: .greedy,
            vocabSize: bundle.vocabSize,
            requestAdmission: requestAdmission
        )
        let resources = ModelResources.shared(for: configuration)

        let tokenizerLoadSpan = InstrumentsProfiler.beginTokenizerLoad(id: bundle.tokenizer)
        let tokenizer = try await bundle.loadTokenizer()
        tokenizerLoadSpan.end()

        // The profile gate. It runs as soon as the tokenizer exists and
        // strictly before the engine is constructed, so an artifact whose
        // template cannot serve the selected profile fails without ever
        // paying for a model load. Nothing below this line may move above it.
        let codec = CoreAITranscriptCodec(profile: protocolProfile)
        try codec.validate(tokenizer: tokenizer)

        if mode == .eager { try await resources.loadResources() }

        self.init(
            configuration: configuration, bundle: bundle, tokenizer: tokenizer,
            protocolProfile: protocolProfile, resources: resources,
            requestAdmission: requestAdmission)
    }

    init(
        configuration: CoreAIExecutor.Configuration,
        bundle: LanguageBundle,
        tokenizer: any Tokenizer,
        protocolProfile: CoreAILanguageProtocolProfile,
        resources: ModelResources,
        requestAdmission: CoreAIRequestAdmission?
    ) {
        self.url = configuration.url
        self.variant = configuration.variant
        self.kvCacheStrategy = configuration.kvCacheStrategy
        self.samplingConfig = configuration.samplingConfig
        self.bundle = bundle
        self.tokenizer = tokenizer
        self.protocolProfile = protocolProfile
        self.codec = CoreAITranscriptCodec(profile: protocolProfile)
        self.resources = resources
        self.requestAdmission = requestAdmission
        // Read additional stop token IDs from tokenizer_config.json (e.g. Gemma's
        // <end_of_turn>). Empty when the bundle has no tokenizer directory.
        if let tokenizerDir = bundle.tokenizerPath {
            self.additionalEosTokenIds = LanguageConfig.additionalStopTokenIds(
                from: tokenizerDir, tokenizer: tokenizer)
        } else {
            self.additionalEosTokenIds = []
        }
    }

    // MARK: - Resource control

    /// Estimated on-disk size of the model's main asset, in bytes.

    public var estimatedSizeOnDiskBytes: Int? {
        guard let assetURL = bundle.modelURL(for: ModelBundle.ComponentKey.main) else { return nil }
        return assetURL.recursiveFileSizeInBytes()
    }

    public func load() async throws {
        try await resources.loadResources()
    }

    public func unload() {
        resources.unloadResources()
    }

    /// Whether guided generation is available for this model.
    private var isGuidedGenerationSupported: Bool {
        if let isConstrainedCapable = resources.loadedEngineIsConstrainedCapable {
            return isConstrainedCapable
        }
        if let supportsLogits = resources.loadedEngineSupportsLogits {
            return supportsLogits
        }
        return true
    }

    // MARK: - Executor

    public struct CoreAIExecutor: LanguageModelExecutor {
        public typealias Model = CoreAILanguageModel

        public struct Configuration: Hashable, Sendable {
            let url: URL
            let variant: String?
            let kvCacheStrategy: KVCacheStrategy
            let modelIdentifier: String
            let samplingConfig: SamplingConfiguration
            let vocabSize: Int?
            let requestAdmission: CoreAIRequestAdmission?
        }

        // MARK: - Properties

        // MARK: - Initialization

        public init(configuration: Configuration) throws {
            _ = configuration
        }

        // MARK: - Prewarm (FoundationModels, synchronous)

        /// Kicks off the engine load in the background.
        public func prewarm(model: CoreAILanguageModel, transcript: Transcript) {
            Task { try? await model.resources.loadResources() }
        }

        // MARK: - respond(to:model:streamingInto:) — new channel-based API

        public nonisolated(nonsending) func respond(
            to request: LanguageModelExecutorGenerationRequest,
            model: CoreAILanguageModel,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            // Tokenization span
            let tokenizationSpan = InstrumentsProfiler.beginTokenization(inputLength: 0)
            let promptTokens: [Int]
            let reasoning: CoreAITranscriptCodec.ReasoningConfiguration
            do {
                // The codec resolves the reasoning intent first — a profile
                // with no suppression mechanism refuses a disable request here
                // rather than quietly generating a thought block anyway.
                reasoning = try model.codec.reasoningConfiguration(
                    for: request.contextOptions.reasoningLevel)
                let encoded = try model.codec.encode(
                    entries: Array(request.transcript),
                    tools: request.enabledToolDefinitions,
                    reasoning: reasoning,
                    using: model.tokenizer)
                try Self.assertTextOnly(encoded, profile: model.protocolProfile)
                promptTokens = encoded.tokens
            } catch {
                tokenizationSpan.end()
                throw error
            }
            tokenizationSpan.end()

            CLILogger.log("Tokenized \(promptTokens.count) tokens", component: "CoreAIExecutor")

            let effectiveSamplingConfig = makeSamplingConfig(
                from: request.generationOptions, base: model.samplingConfig)
            let defaultMaxTokens = model.protocolProfile.supportsReasoning ? 2048 : 512
            let maxTokens = request.generationOptions.maximumResponseTokens ?? defaultMaxTokens

            let metrics = CoreAIRequestMetrics(
                inputTokenCount: promptTokens.count,
                reservedOutputTokenCount: maxTokens,
                attachmentCount: Self.attachmentCount(in: request.transcript))

            // Borrow the engine for the whole generation.
            try await model.resources.withEngine { engine in
                // FoundationModels now threads entry identity itself based on event
                // ordering — we no longer mint an entryID and pass it down.

                switch CoreAIRequestRouting.select(
                    hasSchema: request.schema != nil,
                    enabledToolCount: request.enabledToolDefinitions.count
                ) {
                case .constrainedGeneration:
                    guard let schema = request.schema else {
                        preconditionFailure("Constrained routing requires a response schema")
                    }
                    guard engine.supportsLogits || engine is any ConstrainedGenerationCapable else {
                        throw LanguageModelError.unsupportedCapability(
                            .init(
                                capability: .guidedGeneration,
                                debugDescription:
                                    "This model's inference engine does not support guided generation "
                                    + "(constrained decoding requires per-step logits)."
                            )
                        )
                    }
                    try await respondConstrained(
                        engine: engine,
                        model: model,
                        schema: schema,
                        promptTokens: promptTokens,
                        samplingConfig: effectiveSamplingConfig,
                        maxTokens: maxTokens,
                        metrics: metrics,
                        admission: model.requestAdmission,
                        channel: channel
                    )
                case .protocolGeneration:
                    try await respondVanilla(
                        engine: engine,
                        model: model,
                        promptTokens: promptTokens,
                        samplingConfig: effectiveSamplingConfig,
                        maxTokens: maxTokens,
                        reasoningEnabled: reasoning.enabled,
                        metrics: metrics,
                        admission: model.requestAdmission,
                        channel: channel
                    )
                }
            }
        }

        // MARK: - Vanilla Generation (no schema)

        private func respondVanilla(
            engine: any InferenceEngine,
            model: CoreAILanguageModel,
            promptTokens: [Int],
            samplingConfig: SamplingConfiguration,
            maxTokens: Int,
            reasoningEnabled: Bool,
            metrics: CoreAIRequestMetrics,
            admission: CoreAIRequestAdmission?,
            channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            let tokenizer = model.tokenizer
            let tokenStream = try await CoreAIRequestAdmission.perform(
                metrics: metrics,
                admission: admission
            ) {
                try await engine.generate(
                    with: promptTokens.map(Int32.init),
                    samplingConfiguration: samplingConfig,
                    inferenceOptions: InferenceOptions(maxTokens: maxTokens)
                )
            }

            // All EOS-like tokens: the tokenizer's main EOS plus any additional
            // stop tokens from tokenizer_config.json (e.g. Gemma's <end_of_turn>).
            var eosTokens = Set<Int32>()
            if let id = tokenizer.eosTokenId { eosTokens.insert(Int32(id)) }
            eosTokens.formUnion(model.additionalEosTokenIds)
            // Incremental-decode buffer. After a clean emit, one token is
            // retained as context for the next step (see below). During a
            // multi-byte sequence that hasn't decoded cleanly yet, multiple
            // tokens accumulate until the sequence is complete. In the steady
            // state the buffer holds at most 2 tokens, so tokenizer.decode
            // is O(1) per step.
            var pendingTokens: [Int32] = []
            var previousDecodedText: String = ""
            var tokenStep: Int = 0
            // Segments the decoded stream into response, reasoning and tool
            // call events on the fly. Reasoning content is routed to a
            // top-level `.reasoning(...)` channel event so it lands as its own
            // `Transcript.Reasoning` entry, not mixed into the user-facing
            // `Transcript.Response`. The vocabulary comes from the validated
            // profile, never from the tokenizer's token ids.
            var decoder = CoreAIStreamingOutputDecoder(
                profile: model.protocolProfile,
                reasoningEnabled: reasoningEnabled)
            var generatedTokenCount: Int = 0
            var reasoningTokenCount: Int = 0

            for try await output in tokenStream {
                let token = output.tokenId
                if eosTokens.contains(token) {
                    tokenStream.setStopReason(.eos)
                    break
                }

                pendingTokens.append(token)
                tokenStep += 1
                generatedTokenCount += 1

                let decodeSpan = InstrumentsProfiler.beginDecode(step: tokenStep)
                let decodedText = tokenizer.decode(tokens: pendingTokens.map { Int($0) })
                decodeSpan.end()

                let common = decodedText.commonPrefix(with: previousDecodedText)
                let delta = String(decodedText.dropFirst(common.count))
                // Check for replacement char on the full `decodedText`, not on
                // `delta`. Some tokenizers emit one U+FFFD per attempted decode
                // of an incomplete multi-byte sequence (rather than one per
                // bad byte), so two consecutive partial tokens can produce
                // identical "\u{FFFD}" strings — making `delta` empty and
                // hiding the still-incomplete state. Checking `decodedText`
                // catches that case.
                let hasReplacementChar = decodedText.unicodeScalars.contains { $0 == "\u{FFFD}" }

                if hasReplacementChar {
                    // UTF-8 bytes don't form a clean character yet. Hold the
                    // token and wait for the next iteration to extend the
                    // buffer; don't drop or advance.
                    await channel.send(
                        .response(action: .appendText("", tokenCount: 1))
                    )
                    previousDecodedText = decodedText
                    continue
                }

                for event in try decoder.consume(delta) {
                    if case .reasoning = event { reasoningTokenCount += 1 }
                    await Self.dispatch(event, to: channel)
                }

                // Retain the last token as O(1) context for the next decode.
                // SentencePiece needs at least one prior token to infer the leading
                // ▁ (space) on the following token; clearing to empty decodes each
                // new token in isolation and drops inter-word spaces.
                // Keeping one token bounds re-decode cost to 2 tokens per step.
                // Safe for all supported tokenizers: decode([last]) is a prefix of
                // decode([last, next]) when addPrefixSpace=true (Mistral, Llama, Qwen)
                // and for ByteLevel tokenizers (GPT-2 style) where spaces are direct bytes.
                if let last = pendingTokens.last {
                    pendingTokens = [last]
                    previousDecodedText = tokenizer.decode(tokens: [Int(last)])
                } else {
                    pendingTokens.removeAll(keepingCapacity: true)
                    previousDecodedText = ""
                }
            }

            // Flush the decoder — drains any content held back waiting for a
            // marker. Without this, content right at the EOS boundary would be
            // lost. An unterminated block is a protocol violation only when the
            // model chose to stop; when the token cap, a cancellation, or an
            // engine error cut it off, it is truncation and the partial content
            // is flushed instead of failing the whole response.
            for event in try decoder.finish(truncated: Self.isTruncated(tokenStream.stopReason)) {
                if case .reasoning = event { reasoningTokenCount += 1 }
                await Self.dispatch(event, to: channel)
            }

            await channel.send(
                .response(
                    action: .updateUsage(
                        input: .init(totalTokenCount: promptTokens.count, cachedTokenCount: 0),
                        output: .init(
                            totalTokenCount: generatedTokenCount,
                            reasoningTokenCount: reasoningTokenCount
                        ),
                        metadata: Self.terminalMetadata(for: tokenStream.stopReason)
                    )))

            // Yield to let the engine's tokenSequence Task finish cleanup
            // (putBackEngine, state reset, etc.) before the next respond().
            await Task.yield()
        }

        // MARK: - Stop reason

        /// Whether the generation was cut short by us rather than ended by the
        /// model. Only an end-of-sequence token means the model chose to stop;
        /// `.maxTokens`, `.cancelled` and `.error` are all our budget running
        /// out, and `.stopSequence` is a caller-supplied cut, not an
        /// end-of-turn. A `nil` reason (iteration never ran) is treated as
        /// truncation too — the safe direction, since the strict reading turns
        /// an ordinary short generation into a hard failure.
        static func isTruncated(_ stopReason: StopReason?) -> Bool {
            stopReason != .eos
        }

        private static func terminalMetadata(
            for stopReason: StopReason?
        ) -> [String: any ConvertibleToGeneratedContent] {
            isTruncated(stopReason) ? ["incompleteOutput": true] : [:]
        }

        // MARK: - Event Dispatch

        /// Sends one decoder event on the channel. Shared with the vision
        /// adapter so both adapters stream identically.
        ///
        /// The routing decision itself lives in `CoreAIChannelRouting`, which is
        /// plain `Equatable` data and therefore testable;
        /// `LanguageModelExecutorGenerationChannel.Event` is an opaque struct
        /// with no readable properties, so a test can never inspect what was
        /// sent. Everything that could be wrong — which arm an event takes, and
        /// whether an empty fragment is suppressed — is decided before this
        /// function, leaving three unconditional sends.
        ///
        /// We deliberately do not pass `entryID` — FoundationModels threads
        /// entry identity itself based on event ordering.
        static func dispatch(
            _ event: CoreAIStreamingOutputDecoder.Event,
            to channel: LanguageModelExecutorGenerationChannel
        ) async {
            switch CoreAIChannelRouting(event) {
            case .drop:
                return
            case .appendReasoningText(let text):
                await channel.send(
                    .reasoning(action: .appendText(text, tokenCount: 1))
                )
            case .appendResponseText(let text):
                await channel.send(
                    .response(action: .appendText(text, tokenCount: 1))
                )
            case .appendToolCallArguments(let id, let name, let argumentsJSON):
                // Arguments are model-generated content and are never logged.
                CLILogger.log(
                    "Dispatching tool call id=\(id) name=\(name)",
                    component: "CoreAIExecutor")
                await channel.send(
                    .toolCalls(
                        action: .toolCall(
                            id: id,
                            name: name,
                            action: .appendArguments(argumentsJSON, tokenCount: 1)
                        )
                    )
                )
            }
        }

        // MARK: - Constrained Generation (with schema)

        private func respondConstrained(
            engine: any InferenceEngine,
            model: CoreAILanguageModel,
            schema: GenerationSchema,
            promptTokens: [Int],
            samplingConfig: SamplingConfiguration,
            maxTokens: Int,
            metrics: CoreAIRequestMetrics,
            admission: CoreAIRequestAdmission?,
            channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            let schemaData = try JSONEncoder().encode(schema)

            guard let jsonSchema = String(data: schemaData, encoding: .utf8) else {
                preconditionFailure("GenerationSchema JSON encoding produced invalid UTF-8")
            }

            let stopSequences = StopSequences(
                for: model.tokenizer,
                additionalEosTokenIds: model.additionalEosTokenIds
            )

            let outcome: (generatedTokenCount: Int, stopReason: StopReason?)
            if engine is any ConstrainedGenerationCapable {
                let strategy = PipelinedConstrainedDecodingStrategy(
                    jsonSchema: jsonSchema, vocabSize: model.bundle.vocabSize)
                let stream = try await CoreAIRequestAdmission.perform(
                    metrics: metrics,
                    admission: admission
                ) {
                    try await strategy.decode(
                        from: .tokens(promptTokens),
                        tokenizer: model.tokenizer,
                        inferenceEngine: engine,
                        samplingConfiguration: samplingConfig,
                        options: InferenceOptions(maxTokens: maxTokens),
                        stopSequences: stopSequences
                    )
                }
                outcome = try await Self.forwardConstrained(stream, to: channel)
            } else {
                let strategy = ConstrainedDecodingStrategy(
                    jsonSchema: jsonSchema, vocabSize: model.bundle.vocabSize)
                let stream = try await CoreAIRequestAdmission.perform(
                    metrics: metrics,
                    admission: admission
                ) {
                    try await strategy.decode(
                        from: .tokens(promptTokens),
                        tokenizer: model.tokenizer,
                        inferenceEngine: engine,
                        samplingConfiguration: samplingConfig,
                        options: InferenceOptions(maxTokens: maxTokens),
                        stopSequences: stopSequences
                    )
                }
                outcome = try await Self.forwardConstrained(stream, to: channel)
            }

            await channel.send(
                .response(
                    action: .updateUsage(
                        input: .init(totalTokenCount: promptTokens.count, cachedTokenCount: 0),
                        output: .init(
                            totalTokenCount: outcome.generatedTokenCount,
                            reasoningTokenCount: 0
                        ),
                        metadata: Self.terminalMetadata(for: outcome.stopReason)
                    )))

            // Yield to let the engine's tokenSequence Task finish cleanup
            // (putBackEngine, state reset, etc.) before the next respond().
            await Task.yield()
        }

        private static func forwardConstrained<Sequence>(
            _ stream: Sequence,
            to channel: LanguageModelExecutorGenerationChannel
        ) async throws -> (generatedTokenCount: Int, stopReason: StopReason?)
        where Sequence: StopReasonReportingGenerationSequence {
            var generatedTokenCount = 0
            for try await result in stream {
                generatedTokenCount += 1
                await channel.send(
                    .response(action: .appendText(result.text, tokenCount: 1))
                )
            }
            return (generatedTokenCount, stream.stopReason)
        }

        // MARK: - Transcript inspection

        /// The text adapter has no image pipeline: it forwards only
        /// `EncodedTranscript.tokens` and drops `images`. The codec renders a
        /// prompt holding an attachment as a multi-part `[{"type":"text"},
        /// {"type":"image"}]` content array, which a text-only Jinja template
        /// will happily stringify rather than reject — so an image reaching
        /// this adapter would be silently mis-rendered rather than silently
        /// dropped. Reject it instead. Images belong to
        /// `CoreAIVisionLanguageModel`.
        static func assertTextOnly(
            _ encoded: CoreAITranscriptCodec.EncodedTranscript,
            profile: CoreAILanguageProtocolProfile
        ) throws {
            guard encoded.images.isEmpty else {
                throw CoreAIProtocolError(
                    profile: profile, failure: .unsupportedTranscriptContent)
            }
        }

        private static func attachmentCount(in transcript: Transcript) -> Int {
            transcript.reduce(into: 0) { count, entry in
                let segments: [Transcript.Segment]
                switch entry {
                case .instructions(let value): segments = value.segments
                case .prompt(let value): segments = value.segments
                case .response(let value): segments = value.segments
                case .toolOutput(let value): segments = value.segments
                default: return
                }
                count += segments.reduce(into: 0) { partial, segment in
                    if case .attachment = segment { partial += 1 }
                }
            }
        }

        // MARK: - Helper Methods

        private func makeSamplingConfig(
            from options: GenerationOptions,
            base: SamplingConfiguration
        ) -> SamplingConfiguration {
            if let temperature = options.temperature {
                return SamplingConfiguration(temperature: temperature)
            }
            return base
        }
    }
}

// MARK: - Channel routing

/// Where one `CoreAIStreamingOutputDecoder.Event` goes on a FoundationModels
/// generation channel.
///
/// This exists so the mapping can be asserted. Response text becomes
/// `.response(...).appendText`, reasoning becomes a *top-level*
/// `.reasoning(...).appendText` — a sibling of response and tool calls in this
/// API, not nested under response, because at decode time we do not yet know
/// whether the model will follow a thought block with a response or a tool
/// call — and a completed call becomes `.toolCalls(...).toolCall`. An empty
/// text fragment is dropped rather than sent, matching the behavior of the
/// single-marker hold-back parsers this decoder superseded.
package enum CoreAIChannelRouting: Equatable, Sendable {
    case drop
    case appendResponseText(String)
    case appendReasoningText(String)
    case appendToolCallArguments(id: String, name: String, argumentsJSON: String)

    package init(_ event: CoreAIStreamingOutputDecoder.Event) {
        switch event {
        case .response(let text):
            self = text.isEmpty ? .drop : .appendResponseText(text)
        case .reasoning(let text):
            self = text.isEmpty ? .drop : .appendReasoningText(text)
        case .toolCall(let id, let name, let argumentsJSON):
            self = .appendToolCallArguments(id: id, name: name, argumentsJSON: argumentsJSON)
        }
    }
}
