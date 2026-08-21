// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation
import FoundationModels
import ImageIO
import Synchronization
import TestUtilities
import Testing
import Tokenizers

@testable import CoreAILanguageModels

@Suite("Core AI request admission")
struct RequestAdmissionTests {
    private enum ProbeError: Error {
        case rejected
    }

    private actor GenerationProbe {
        private(set) var starts = 0

        func start() {
            starts += 1
        }
    }

    @Test("rejection prevents the generation operation")
    func rejectionPreventsGeneration() async {
        let probe = GenerationProbe()
        let metrics = CoreAIRequestMetrics(
            inputTokenCount: 41,
            reservedOutputTokenCount: 17,
            attachmentCount: 1)
        let admission = CoreAIRequestAdmission { received in
            #expect(received == metrics)
            throw ProbeError.rejected
        }

        await #expect(throws: ProbeError.self) {
            _ = try await CoreAIRequestAdmission.perform(
                metrics: metrics,
                admission: admission
            ) {
                await probe.start()
                return ()
            }
        }
        #expect(await probe.starts == 0)
    }

    @Test("cancellation propagates while admission is running")
    func cancellationDuringAdmission() async {
        let probe = GenerationProbe()
        let (started, startedContinuation) = AsyncStream<Void>.makeStream()
        let task = Task {
            _ = try await CoreAIRequestAdmission.perform(
                metrics: .init(
                    inputTokenCount: 1,
                    reservedOutputTokenCount: 1,
                    attachmentCount: 0),
                admission: CoreAIRequestAdmission { _ in
                    startedContinuation.yield()
                    startedContinuation.finish()
                    try await Task.sleep(for: .seconds(60))
                }
            ) {
                await probe.start()
                return ()
            }
        }

        for await _ in started { break }
        task.cancel()
        await #expect(throws: CancellationError.self) {
            try await task.value
        }
        #expect(await probe.starts == 0)
    }

    @Test("cancellation propagates after generation starts")
    func cancellationDuringGeneration() async {
        let (started, startedContinuation) = AsyncStream<Void>.makeStream()
        let task = Task {
            try await CoreAIRequestAdmission.perform(
                metrics: .init(
                    inputTokenCount: 1,
                    reservedOutputTokenCount: 1,
                    attachmentCount: 0),
                admission: CoreAIRequestAdmission { _ in }
            ) {
                startedContinuation.yield()
                startedContinuation.finish()
                try await Task.sleep(for: .seconds(60))
            }
        }

        for await _ in started { break }
        task.cancel()
        await #expect(throws: CancellationError.self) {
            try await task.value
        }
    }

    @Test("metrics retain the authoritative input count used for usage")
    func metricsMatchUsageInput() async throws {
        let usageInputCount = 73
        let admission = CoreAIRequestAdmission { metrics in
            #expect(metrics.inputTokenCount == usageInputCount)
            #expect(metrics.reservedOutputTokenCount == 29)
            #expect(metrics.attachmentCount == 2)
        }
        try await admission(
            .init(
                inputTokenCount: usageInputCount,
                reservedOutputTokenCount: 29,
                attachmentCount: 2))
    }
}

@Suite("Core AI reasoning policy")
struct ReasoningPolicyTests {
    /// Counts calls across the `Sendable` tokenizer stubs below without
    /// needing a non-copyable lock inside a copyable struct.
    private final class CallRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0

        func record() {
            lock.lock()
            count += 1
            lock.unlock()
        }

        var value: Int {
            lock.lock()
            defer { lock.unlock() }
            return count
        }
    }

    private struct TemplateFailure: Error {}

    /// Advertises a chat template and then refuses to render it. `encodes`
    /// counts every plain-text `encode(text:)` — precisely the call the
    /// deleted concatenation fallback made when the template threw.
    private struct RefusingTemplateTokenizer: Tokenizer, Sendable {
        let encodes: CallRecorder

        var hasChatTemplate: Bool { true }
        var bosToken: String? { nil }
        var bosTokenId: Int? { nil }
        var eosToken: String? { nil }
        var eosTokenId: Int? { nil }
        var unknownToken: String? { nil }
        var unknownTokenId: Int? { nil }

        func tokenize(text: String) -> [String] { [text] }
        func encode(text: String) -> [Int] {
            encodes.record()
            return [1, 2, 3]
        }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { encode(text: text) }
        func callAsFunction(_ text: String, addSpecialTokens: Bool) -> [Int] { encode(text: text) }
        func decode(tokens: [Int]) -> String { "" }
        func decode(tokens: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertTokensToIds(_ tokens: [String]) -> [Int?] { tokens.map { _ in nil } }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func convertIdsToTokens(_ ids: [Int]) -> [String?] { ids.map { _ in nil } }

        func applyChatTemplate(messages: [Message]) throws -> [Int] { throw TemplateFailure() }
        func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] {
            throw TemplateFailure()
        }
        func applyChatTemplate(
            messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            throw TemplateFailure()
        }
        func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] {
            throw TemplateFailure()
        }
        func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
            throw TemplateFailure()
        }
        func applyChatTemplate(
            messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool,
            truncation: Bool, maxLength: Int?, tools: [ToolSpec]?
        ) throws -> [Int] {
            throw TemplateFailure()
        }
        func applyChatTemplate(
            messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool,
            truncation: Bool, maxLength: Int?, tools: [ToolSpec]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            throw TemplateFailure()
        }
    }

    private static let syntheticPrompt: [Transcript.Entry] = [
        .prompt(Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: "synthetic-user"))]))
    ]

    @Test("An unrequested reasoning level defers to the profile's own default")
    func defaultsFollowTheProfile() throws {
        for profile in CoreAILanguageProtocolProfile.allCases {
            let codec = CoreAITranscriptCodec(profile: profile)
            let configuration = try codec.reasoningConfiguration(for: nil)
            #expect(
                configuration.enabled == profile.defaultReasoningEnabled,
                "\(profile.rawValue) ignored its own default")
        }
    }

    @Test("A profile with no suppression mechanism refuses to pretend it disabled reasoning")
    func unsuppressibleProfilesRejectDisabling() {
        for profile in CoreAILanguageProtocolProfile.allCases
        where profile.supportsReasoning && !profile.supportsDisablingReasoning {
            let codec = CoreAITranscriptCodec(profile: profile)
            #expect(
                throws: CoreAIProtocolError(profile: profile, failure: .unsupportedReasoningPolicy),
                "\(profile.rawValue) silently accepted a disable it cannot honor"
            ) {
                _ = try codec.reasoningConfiguration(for: .custom("no_think"))
            }
        }
    }

    @Test("A suppressible profile disables reasoning and says so to the template")
    func suppressibleProfilesHonorDisabling() throws {
        for profile in CoreAILanguageProtocolProfile.allCases
        where profile.supportsDisablingReasoning {
            let codec = CoreAITranscriptCodec(profile: profile)
            let configuration = try codec.reasoningConfiguration(for: .custom("no_think"))
            #expect(!configuration.enabled, "\(profile.rawValue) stayed enabled")
            #expect(configuration.additionalContext?["enable_thinking"] as? Bool == false)
        }
    }

    @Test("A profile without reasoning rejects every reasoning level")
    func plainChatRejectsEveryReasoningLevel() {
        let codec = CoreAITranscriptCodec(profile: .plainChat)
        let levels: [ContextOptions.ReasoningLevel] = [
            .light, .moderate, .deep, .custom("no_think"),
        ]
        for level in levels {
            #expect(
                throws: CoreAIProtocolError(profile: .plainChat, failure: .unsupportedReasoningPolicy)
            ) {
                _ = try codec.reasoningConfiguration(for: level)
            }
        }
    }

    @Test("An unknown reasoning policy is rejected rather than approximated")
    func unknownReasoningPolicyRejected() {
        let codec = CoreAITranscriptCodec(profile: .qwen35XML)
        #expect(
            throws: CoreAIProtocolError(profile: .qwen35XML, failure: .unsupportedReasoningPolicy)
        ) {
            _ = try codec.reasoningConfiguration(for: .custom("unsupported"))
        }
    }

    @Test("A template that refuses the transcript never falls back to plain text")
    func templateFailureNeverFallsBackToPlainText() throws {
        let encodes = CallRecorder()
        let tokenizer = RefusingTemplateTokenizer(encodes: encodes)
        let codec = CoreAITranscriptCodec(profile: .plainChat)
        #expect(
            throws: CoreAIProtocolError(profile: .plainChat, failure: .incompatibleChatTemplate)
        ) {
            _ = try codec.encode(
                entries: Self.syntheticPrompt,
                tools: [],
                reasoning: try codec.reasoningConfiguration(for: nil),
                using: tokenizer)
        }
        // The deleted fallback joined the message contents and called
        // `encode(text:)`. Reintroducing it makes this count non-zero.
        #expect(encodes.value == 0, "the transcript was encoded as plain text after all")
    }
}

@Suite("Text adapter transcript limits")
struct TextAdapterTranscriptTests {
    /// Renders any transcript without complaint, so the assertion under test
    /// is the adapter's own and not the template's.
    private struct RenderingTokenizer: Tokenizer, Sendable {
        var hasChatTemplate: Bool { true }
        var bosToken: String? { nil }
        var bosTokenId: Int? { nil }
        var eosToken: String? { nil }
        var eosTokenId: Int? { nil }
        var unknownToken: String? { nil }
        var unknownTokenId: Int? { nil }

        func tokenize(text: String) -> [String] { [text] }
        func encode(text: String) -> [Int] { [1, 2, 3] }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1, 2, 3] }
        func callAsFunction(_ text: String, addSpecialTokens: Bool) -> [Int] { [1, 2, 3] }
        func decode(tokens: [Int]) -> String { "" }
        func decode(tokens: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertTokensToIds(_ tokens: [String]) -> [Int?] { tokens.map { _ in nil } }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func convertIdsToTokens(_ ids: [Int]) -> [String?] { ids.map { _ in nil } }

        func applyChatTemplate(messages: [Message]) throws -> [Int] { [1, 2, 3] }
        func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] { [1, 2, 3] }
        func applyChatTemplate(
            messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [1, 2, 3] }
        func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] {
            [1, 2, 3]
        }
        func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] { [1, 2, 3] }
        func applyChatTemplate(
            messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool,
            truncation: Bool, maxLength: Int?, tools: [ToolSpec]?
        ) throws -> [Int] { [1, 2, 3] }
        func applyChatTemplate(
            messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool,
            truncation: Bool, maxLength: Int?, tools: [ToolSpec]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [1, 2, 3] }
    }

    private static func encode(
        segments: [Transcript.Segment]
    ) throws -> CoreAITranscriptCodec.EncodedTranscript {
        let codec = CoreAITranscriptCodec(profile: .plainChat)
        return try codec.encode(
            entries: [.prompt(Transcript.Prompt(segments: segments))],
            tools: [],
            reasoning: try codec.reasoningConfiguration(for: nil),
            using: RenderingTokenizer())
    }

    @Test("A text-only transcript is accepted")
    func textOnlyTranscriptAccepted() throws {
        let encoded = try Self.encode(
            segments: [.text(Transcript.TextSegment(content: "synthetic-user"))])
        #expect(encoded.images.isEmpty)
        try CoreAILanguageModel.CoreAIExecutor.assertTextOnly(encoded, profile: .plainChat)
    }

    @Test("An image reaching the text adapter is rejected, not rendered as a content array")
    func imageInTextTranscriptRejected() throws {
        let cgImage = try #require(makeSolidCGImage(r: 10, g: 20, b: 30, side: 2))
        let encoded = try Self.encode(segments: [
            .text(Transcript.TextSegment(content: "synthetic-user")),
            .attachment(
                Transcript.AttachmentSegment(
                    content: .image(Transcript.ImageAttachment(cgImage)))),
        ])
        // The codec turns an attachment into a multi-part
        // [{"type":"text"},{"type":"image"}] content array. A text-only Jinja
        // template stringifies that rather than throwing, so without this
        // guard the image is silently mis-rendered into the prompt.
        #expect(encoded.images.count == 1)
        #expect(
            throws: CoreAIProtocolError(
                profile: .plainChat, failure: .unsupportedTranscriptContent)
        ) {
            try CoreAILanguageModel.CoreAIExecutor.assertTextOnly(encoded, profile: .plainChat)
        }
    }
}

@Suite("Core AI image orientation")
struct CoreAIImageOrientationTests {
    @Test("the VLM accepts exactly one image")
    func exactlyOneImage() throws {
        try CoreAIVLMExecutor.validateImageCount(1)
        #expect(throws: CoreAIVisionRequestError.requiresExactlyOneImage(actualCount: 0)) {
            try CoreAIVLMExecutor.validateImageCount(0)
        }
        #expect(throws: CoreAIVisionRequestError.requiresExactlyOneImage(actualCount: 2)) {
            try CoreAIVLMExecutor.validateImageCount(2)
        }
    }

    @Test(
        "all EXIF orientations render upright",
        arguments: [
            CGImagePropertyOrientation.up,
            .upMirrored,
            .down,
            .downMirrored,
            .leftMirrored,
            .right,
            .rightMirrored,
            .left,
        ])
    func rendersUpright(orientation: CGImagePropertyOrientation) throws {
        let image = try #require(makeImage(width: 3, height: 2))
        let attachment = Transcript.ImageAttachment(image, orientation: orientation)
        let rendered = try CoreAIVLMExecutor.uprightCGImage(from: attachment)
        let swapsAxes: Bool
        switch orientation {
        case .leftMirrored, .right, .rightMirrored, .left:
            swapsAxes = true
        default:
            swapsAxes = false
        }
        #expect(rendered.width == (swapsAxes ? 2 : 3))
        #expect(rendered.height == (swapsAxes ? 3 : 2))
    }

    private func makeImage(width: Int, height: Int) -> CGImage? {
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
            let context = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()
    }
}
