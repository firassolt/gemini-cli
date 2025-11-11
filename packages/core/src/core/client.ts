/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  GenerateContentConfig,
  PartListUnion,
  Content,
  GenerateContentResponse,
} from '@google/genai';
import type { ServerGeminiStreamEvent, ChatCompressionInfo } from './turn.js';
import type {
  ChatRecordingService,
  ResumedSessionData,
} from '../services/chatRecordingService.js';
import type { GeminiChat } from './geminiChat.js';
import type { IdeContext } from '../ide/types.js';
import type { Config } from '../config/config.js';
import { ConversationLifecycleService } from './conversationLifecycleService.js';
import { LoopDetectionService } from '../services/loopDetectionService.js';
import { ChatCompressionService } from '../services/chatCompressionService.js';
import { Turn } from './turn.js';

export {
  ConversationLifecycleService,
  isThinkingSupported,
  isThinkingDefault,
} from './conversationLifecycleService.js';

export class GeminiClient {
  private readonly lifecycleService: ConversationLifecycleService;
  private readonly config: Config;
  // The following declarations maintain compatibility with test suites that
  // reach into the Gemini client to stub internal collaborators.
  private declare chat?: GeminiChat;
  private declare startChat: (
    ...args: unknown[]
  ) => Promise<GeminiChat> | GeminiChat;
  private declare currentSequenceModel: string | null;
  private declare forceFullIdeContext: boolean;
  private declare lastSentIdeContext?: IdeContext;
  private declare loopDetector: LoopDetectionService;

  constructor(
    config: Config,
    lifecycleService?: ConversationLifecycleService,
  ) {
    this.config = config;
    this.lifecycleService =
      lifecycleService ??
      new ConversationLifecycleService(
        config,
        new LoopDetectionService(config),
        new ChatCompressionService(),
      );

    // Maintain access to the configuration object for tests that mock internals.
    void this.config;

    const lifecycle: Record<string, unknown> =
      this.lifecycleService as unknown as Record<string, unknown>;

    Object.defineProperties(this, {
      chat: {
        configurable: true,
        get: () => lifecycle['chat'],
        set: (value) => {
          lifecycle['chat'] = value;
        },
      },
      startChat: {
        configurable: true,
        get: () => {
          const startChat = lifecycle['startChat'];
          if (typeof startChat === 'function') {
            return startChat.bind(this.lifecycleService);
          }
          return startChat;
        },
        set: (value) => {
          lifecycle['startChat'] = value;
        },
      },
      currentSequenceModel: {
        configurable: true,
        get: () => lifecycle['currentSequenceModel'],
        set: (value) => {
          lifecycle['currentSequenceModel'] = value;
        },
      },
      forceFullIdeContext: {
        configurable: true,
        get: () => lifecycle['forceFullIdeContext'],
        set: (value) => {
          lifecycle['forceFullIdeContext'] = value;
        },
      },
      lastSentIdeContext: {
        configurable: true,
        get: () => lifecycle['lastSentIdeContext'],
        set: (value) => {
          lifecycle['lastSentIdeContext'] = value;
        },
      },
      loopDetector: {
        configurable: true,
        get: () => lifecycle['loopDetector'],
      },
    });
  }

  async initialize(): Promise<void> {
    await this.lifecycleService.initialize();
  }

  async addHistory(content: Content): Promise<void> {
    await this.lifecycleService.addHistory(content);
  }

  getChat(): GeminiChat {
    return this.lifecycleService.getChat();
  }

  isInitialized(): boolean {
    return this.lifecycleService.isInitialized();
  }

  getHistory(): Content[] {
    return this.lifecycleService.getHistory();
  }

  stripThoughtsFromHistory(): void {
    this.lifecycleService.stripThoughtsFromHistory();
  }

  setHistory(history: Content[]): void {
    this.lifecycleService.setHistory(history);
  }

  async setTools(): Promise<void> {
    await this.lifecycleService.setTools();
  }

  async resetChat(): Promise<void> {
    await this.lifecycleService.resetChat();
  }

  async resumeChat(
    history: Content[],
    resumedSessionData?: ResumedSessionData,
  ): Promise<void> {
    await this.lifecycleService.resumeChat(history, resumedSessionData);
  }

  getChatRecordingService(): ChatRecordingService | undefined {
    return this.lifecycleService.getChatRecordingService();
  }

  getLoopDetectionService(): LoopDetectionService {
    return this.lifecycleService.getLoopDetectionService();
  }

  getCurrentSequenceModel(): string | null {
    return this.lifecycleService.getCurrentSequenceModel();
  }

  async addDirectoryContext(): Promise<void> {
    await this.lifecycleService.addDirectoryContext();
  }

  async *sendMessageStream(
    request: PartListUnion,
    signal: AbortSignal,
    prompt_id: string,
    turns?: number,
    isInvalidStreamRetry?: boolean,
  ): AsyncGenerator<ServerGeminiStreamEvent, Turn> {
    return yield* this.lifecycleService.sendMessageStream(
      request,
      signal,
      prompt_id,
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
    return this.lifecycleService.generateContent(
      contents,
      generationConfig,
      abortSignal,
      model,
    );
  }

  async tryCompressChat(
    prompt_id: string,
    force?: boolean,
  ): Promise<ChatCompressionInfo> {
    return this.lifecycleService.tryCompressChat(prompt_id, force);
  }
}
