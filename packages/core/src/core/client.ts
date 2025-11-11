/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  Content,
  GenerateContentConfig,
  GenerateContentResponse,
  PartListUnion,
} from '@google/genai';

import type {
  ChatRecordingService,
  ResumedSessionData,
} from '../services/chatRecordingService.js';
import type { GeminiChat } from './geminiChat.js';
import type { Config } from '../config/config.js';
import type { ChatCompressionInfo, ServerGeminiStreamEvent } from './turn.js';
import { Turn } from './turn.js';
import { ConversationLifecycleService } from './conversationLifecycleService.js';
import { LoopDetectionService } from '../services/loopDetectionService.js';
import { ChatCompressionService } from '../services/chatCompressionService.js';

export { ConversationLifecycleService, isThinkingDefault, isThinkingSupported } from './conversationLifecycleService.js';

type GeminiClientDependencies = {
  lifecycleService?: ConversationLifecycleService;
  loopDetectionService?: LoopDetectionService;
  chatCompressionService?: ChatCompressionService;
};

export class GeminiClient {
  private readonly lifecycle: ConversationLifecycleService;

  constructor(
    readonly config: Config,
    dependencies: GeminiClientDependencies = {},
  ) {
    if (dependencies.lifecycleService) {
      this.lifecycle = dependencies.lifecycleService;
    } else {
      const loopDetector =
        dependencies.loopDetectionService ?? new LoopDetectionService(this.config);
      const compressionService =
        dependencies.chatCompressionService ?? new ChatCompressionService();
      this.lifecycle = new ConversationLifecycleService(
        this.config,
        loopDetector,
        compressionService,
      );
    }
  }

  getLifecycleForTesting(): ConversationLifecycleService {
    return this.lifecycle;
  }

  async initialize(): Promise<void> {
    await this.lifecycle.initialize();
  }

  async addHistory(content: Content): Promise<void> {
    this.lifecycle.addHistory(content);
  }

  getChat(): GeminiChat {
    return this.lifecycle.getChat();
  }

  isInitialized(): boolean {
    return this.lifecycle.isInitialized();
  }

  getHistory(): Content[] {
    return this.lifecycle.getHistory();
  }

  stripThoughtsFromHistory(): void {
    this.lifecycle.stripThoughtsFromHistory();
  }

  setHistory(history: Content[]): void {
    this.lifecycle.setHistory(history);
  }

  async setTools(): Promise<void> {
    await this.lifecycle.setTools();
  }

  async resetChat(): Promise<void> {
    await this.lifecycle.resetChat();
  }

  async resumeChat(
    history: Content[],
    resumedSessionData?: ResumedSessionData,
  ): Promise<void> {
    await this.lifecycle.resumeChat(history, resumedSessionData);
  }

  getChatRecordingService(): ChatRecordingService | undefined {
    return this.lifecycle.getChatRecordingService();
  }

  getLoopDetectionService(): LoopDetectionService {
    return this.lifecycle.getLoopDetectionService();
  }

  getCurrentSequenceModel(): string | null {
    return this.lifecycle.getCurrentSequenceModel();
  }

  async addDirectoryContext(): Promise<void> {
    await this.lifecycle.addDirectoryContext();
  }

  async *sendMessageStream(
    request: PartListUnion,
    signal: AbortSignal,
    promptId: string,
    turns?: number,
    isInvalidStreamRetry?: boolean,
  ): AsyncGenerator<ServerGeminiStreamEvent, Turn> {
    return yield* this.lifecycle.sendMessageStream(
      request,
      signal,
    };

    let modelToUse: string;

    // Determine Model (Stickiness vs. Routing)
    if (this.currentSequenceModel) {
      modelToUse = this.currentSequenceModel;
    } else {
      const router = await this.config.getModelRouterService();
      const decision = await router.route(routingContext);
      modelToUse = decision.model;
      // Lock the model for the rest of the sequence
      this.currentSequenceModel = modelToUse;
    }

    const resultStream = turn.run(modelToUse, request, linkedSignal);
    for await (const event of resultStream) {
      if (this.loopDetector.addAndCheck(event)) {
        yield { type: GeminiEventType.LoopDetected };
        controller.abort();
        return turn;
      }
      yield event;

      this.updateTelemetryTokenCount();

      if (event.type === GeminiEventType.InvalidStream) {
        const invalidCategory = event.value?.category;
        if (
          this.config.getContinueOnFailedApiCall() &&
          invalidCategory !== 'SAFETY_BLOCK'
        ) {
          if (isInvalidStreamRetry) {
            // We already retried once, so stop here.
            logContentRetryFailure(
              this.config,
              new ContentRetryFailureEvent(
                4, // 2 initial + 2 after injections
                'FAILED_AFTER_PROMPT_INJECTION',
                modelToUse,
              ),
            );
            return turn;
          }
          const nextRequest = [{ text: 'System: Please continue.' }];
          yield* this.sendMessageStream(
            nextRequest,
            signal,
            prompt_id,
            boundedTurns - 1,
            true, // Set isInvalidStreamRetry to true
          );
          return turn;
        }
      }
      if (event.type === GeminiEventType.Error) {
        return turn;
      }
    }
    if (!turn.pendingToolCalls.length && signal && !signal.aborted) {
      // Check if next speaker check is needed
      if (this.config.getQuotaErrorOccurred()) {
        return turn;
      }

      if (this.config.getSkipNextSpeakerCheck()) {
        return turn;
      }

      const nextSpeakerCheck = await checkNextSpeaker(
        this.getChat(),
        this.config.getBaseLlmClient(),
        signal,
        prompt_id,
      );
      logNextSpeakerCheck(
        this.config,
        new NextSpeakerCheckEvent(
          prompt_id,
          turn.finishReason?.toString() || '',
          nextSpeakerCheck?.next_speaker || '',
        ),
      );
      if (nextSpeakerCheck?.next_speaker === 'model') {
        const nextRequest = [{ text: 'Please continue.' }];
        // This recursive call's events will be yielded out, but the final
        // turn object will be from the top-level call.
        yield* this.sendMessageStream(
          nextRequest,
          signal,
          prompt_id,
          boundedTurns - 1,
          // isInvalidStreamRetry is false here, as this is a next speaker check
        );
      }
    }
    return turn;
  }

  async generateContent(
    contents: Content[],
    generationConfig: GenerateContentConfig,
    abortSignal: AbortSignal,
    model: string,
  ): Promise<GenerateContentResponse> {
    return this.lifecycle.generateContent(
      contents,
      generationConfig,
      abortSignal,
      model,
    );
  }

  async tryCompressChat(
    promptId: string,
    force?: boolean,
  ): Promise<ChatCompressionInfo> {
    return this.lifecycle.tryCompressChat(promptId, force);
  }
}

