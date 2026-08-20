// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
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

@Suite("Core AI reasoning markers")
struct ReasoningMarkerTests {
    private enum TemplateFailure: Error {
        case rejected
    }

    @Test("only a complete marker pair advertises reasoning")
    func completePairRequired() {
        let complete = MockTokenizer(vocab: ["<think>": 10, "</think>": 11])
        let openingOnly = MockTokenizer(vocab: ["<think>": 10])
        let closingOnly = MockTokenizer(vocab: ["</think>": 11])

        #expect(CoreAILanguageModel.CoreAIExecutor.detectThinkingMarkers(using: complete) != nil)
        #expect(CoreAILanguageModel.CoreAIExecutor.detectThinkingMarkers(using: openingOnly) == nil)
        #expect(CoreAILanguageModel.CoreAIExecutor.detectThinkingMarkers(using: closingOnly) == nil)
    }

    @Test("automatic reasoning does not set a template flag")
    func automaticReasoningTemplateContext() throws {
        let observedContext = Mutex<Bool?>(nil)
        let tokenizer = MockTokenizer(additionalContextObserver: { context in
            observedContext.withLock { $0 = context != nil }
        })
        let context = try CoreAILanguageModel.CoreAIExecutor.reasoningTemplateContext(
            nil,
            supportsReasoning: true)

        _ = try CoreAILanguageModel.CoreAIExecutor.applyChatTemplate(
            messages: [["role": "user", "content": "synthetic"] as Message],
            tools: nil,
            using: tokenizer,
            additionalContext: context)

        #expect(observedContext.withLock { $0 } == false)
    }

    @Test("disabled reasoning passes enable_thinking false to the tokenizer")
    func disabledReasoningTemplateContext() throws {
        let observedContext = Mutex<Bool?>(nil)
        let observedFlag = Mutex<Bool?>(nil)
        let tokenizer = MockTokenizer(
            vocab: ["<think>": 10, "</think>": 11],
            additionalContextObserver: { context in
                observedContext.withLock { $0 = context != nil }
                observedFlag.withLock { $0 = context?["enable_thinking"] as? Bool }
            })
        let context = try CoreAILanguageModel.CoreAIExecutor.reasoningTemplateContext(
            .custom("no_think"),
            supportsReasoning: true)

        _ = try CoreAILanguageModel.CoreAIExecutor.applyChatTemplate(
            messages: [["role": "user", "content": "synthetic"] as Message],
            tools: nil,
            using: tokenizer,
            additionalContext: context)

        #expect(observedContext.withLock { $0 } == true)
        #expect(observedFlag.withLock { $0 } == false)
    }

    @Test("unknown reasoning policies are rejected")
    func unknownReasoningPolicy() {
        #expect(throws: LanguageModelError.self) {
            _ = try CoreAILanguageModel.CoreAIExecutor.reasoningTemplateContext(
                .custom("unsupported"),
                supportsReasoning: true)
        }
    }

    @Test("disabled reasoning never falls back when the template rejects its flag")
    func disabledReasoningTemplateFailure() throws {
        let tokenizer = MockTokenizer(additionalContextObserver: { _ in
            throw TemplateFailure.rejected
        })
        let context = try CoreAILanguageModel.CoreAIExecutor.reasoningTemplateContext(
            .custom("no_think"),
            supportsReasoning: true)

        #expect(throws: LanguageModelError.self) {
            _ = try CoreAILanguageModel.CoreAIExecutor.applyChatTemplate(
                messages: [["role": "user", "content": "synthetic"] as Message],
                tools: nil,
                using: tokenizer,
                additionalContext: context)
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
