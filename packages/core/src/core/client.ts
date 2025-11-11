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
      promptId,
      turns,
      isInvalidStreamRetry,
    );
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

