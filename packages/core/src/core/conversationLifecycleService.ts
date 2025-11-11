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
  Tool,
} from '@google/genai';

import type { Config } from '../config/config.js';
import {
  DEFAULT_GEMINI_FLASH_MODEL,
  DEFAULT_GEMINI_MODEL,
  DEFAULT_GEMINI_MODEL_AUTO,
  DEFAULT_THINKING_MODE,
  getEffectiveModel,
} from '../config/models.js';
import { handleFallback } from '../fallback/handler.js';
import type { IdeContext, File } from '../ide/types.js';
import { ideContextStore } from '../ide/ideContext.js';
import type {
  ChatRecordingService,
  ResumedSessionData,
} from '../services/chatRecordingService.js';
import type { ChatCompressionService } from '../services/chatCompressionService.js';
import type { LoopDetectionService } from '../services/loopDetectionService.js';
import type { ContentGenerator } from './contentGenerator.js';
import { GeminiChat } from './geminiChat.js';
import { getCoreSystemPrompt } from './prompts.js';
import { GeminiEventType, Turn, CompressionStatus } from './turn.js';
import type { ChatCompressionInfo, ServerGeminiStreamEvent } from './turn.js';
import { tokenLimit } from './tokenLimits.js';
import { reportError } from '../utils/errorReporting.js';
import { retryWithBackoff } from '../utils/retry.js';
import { getErrorMessage } from '../utils/errors.js';
import {
  getDirectoryContextString,
  getInitialChatHistory,
} from '../utils/environmentContext.js';
import { uiTelemetryService } from '../telemetry/uiTelemetry.js';
import {
  logContentRetryFailure,
  logNextSpeakerCheck,
} from '../telemetry/loggers.js';
import {
  ContentRetryFailureEvent,
  NextSpeakerCheckEvent,
} from '../telemetry/types.js';
import { checkNextSpeaker } from '../utils/nextSpeakerChecker.js';
import { debugLogger } from '../utils/debugLogger.js';
import type { RoutingContext } from '../routing/routingStrategy.js';

const MAX_TURNS = 100;

export function isThinkingSupported(model: string) {
  return model.startsWith('gemini-2.5') || model === DEFAULT_GEMINI_MODEL_AUTO;
}

export function isThinkingDefault(model: string) {
  if (model.startsWith('gemini-2.5-flash-lite')) {
    return false;
  }
  return model.startsWith('gemini-2.5') || model === DEFAULT_GEMINI_MODEL_AUTO;
}

type ContextUpdate = {
  contextParts: string[];
  newIdeContext: IdeContext | undefined;
};

export class ConversationLifecycleService {
  private chat?: GeminiChat;
  private readonly baseGenerateConfig: GenerateContentConfig = {
    temperature: 0,
    topP: 1,
  };
  private sessionTurnCount = 0;
  private lastPromptId: string;
  private currentSequenceModel: string | null = null;
  private lastSentIdeContext: IdeContext | undefined;
  private forceFullIdeContext = true;
  private hasFailedCompressionAttempt = false;

  constructor(
    private readonly config: Config,
    private readonly loopDetector: LoopDetectionService,
    private readonly compressionService: ChatCompressionService,
  ) {
    this.lastPromptId = this.config.getSessionId();
  }

  getLoopDetectionService(): LoopDetectionService {
    return this.loopDetector;
  }

  getChatCompressionService(): ChatCompressionService {
    return this.compressionService;
  }

  getConfig(): Config {
    return this.config;
  }

  incrementSessionTurnCount(): number {
    this.sessionTurnCount += 1;
    return this.sessionTurnCount;
  }

  getLastPromptId(): string {
    return this.lastPromptId;
  }

  setLastPromptId(promptId: string): void {
    this.lastPromptId = promptId;
  }

  getCurrentSequenceModel(): string | null {
    return this.currentSequenceModel;
  }

  setCurrentSequenceModel(model: string | null): void {
    this.currentSequenceModel = model;
  }

  markIdeContextSent(context: IdeContext | undefined): void {
    this.lastSentIdeContext = context;
    this.forceFullIdeContext = false;
  }

  forceFullIdeContextUpdate(): void {
    this.forceFullIdeContext = true;
  }

  getLastSentIdeContext(): IdeContext | undefined {
    return this.lastSentIdeContext;
  }

  hasCompressionFailure(): boolean {
    return this.hasFailedCompressionAttempt;
  }

  markCompressionFailure(): void {
    this.hasFailedCompressionAttempt = true;
  }

  clearCompressionFailure(): void {
    this.hasFailedCompressionAttempt = false;
  }

  getChat(): GeminiChat {
    if (!this.chat) {
      throw new Error('Chat not initialized');
    }
    return this.chat;
  }

  isInitialized(): boolean {
    return this.chat !== undefined;
  }

  async initialize(): Promise<void> {
    this.chat = await this.startChat();
    this.updateTelemetryTokenCount();
  }

  async resetChat(): Promise<void> {
    this.chat = await this.startChat();
    this.updateTelemetryTokenCount();
    this.forceFullIdeContextUpdate();
  }

  async resumeChat(
    history: Content[],
    resumedSessionData?: ResumedSessionData,
  ): Promise<void> {
    this.chat = await this.startChat(history, resumedSessionData);
    this.forceFullIdeContextUpdate();
  }

  getChatRecordingService(): ChatRecordingService | undefined {
    return this.chat?.getChatRecordingService();
  }

  addHistory(content: Content): void {
    this.getChat().addHistory(content);
  }

  getHistory(): Content[] {
    return this.getChat().getHistory();
  }

  stripThoughtsFromHistory(): void {
    this.getChat().stripThoughtsFromHistory();
  }

  setHistory(history: Content[]): void {
    this.getChat().setHistory(history);
    this.forceFullIdeContextUpdate();
  }

  async setTools(): Promise<void> {
    const toolRegistry = this.config.getToolRegistry();
    const toolDeclarations = toolRegistry.getFunctionDeclarations();
    const tools: Tool[] = [{ functionDeclarations: toolDeclarations }];
    this.getChat().setTools(tools);
  }

  async addDirectoryContext(): Promise<void> {
    if (!this.chat) {
      return;
    }

    this.getChat().addHistory({
      role: 'user',
      parts: [{ text: await getDirectoryContextString(this.config) }],
    });
  }

  async tryCompressChat(
    promptId: string,
    force: boolean = false,
  ): Promise<ChatCompressionInfo> {
    const model = this.getEffectiveModelForCurrentTurn();
    const { newHistory, info } = await this.compressionService.compress(
      this.getChat(),
      promptId,
      force,
      model,
      this.config,
      this.hasCompressionFailure(),
    );

    if (
      info.compressionStatus ===
      CompressionStatus.COMPRESSION_FAILED_INFLATED_TOKEN_COUNT
    ) {
      if (force) {
        this.clearCompressionFailure();
      } else {
        this.markCompressionFailure();
      }
    }

    if (info.compressionStatus === CompressionStatus.COMPRESSED && newHistory) {
      this.chat = await this.startChat(newHistory);
      this.updateTelemetryTokenCount();
      this.forceFullIdeContextUpdate();
    }

    return info;
  }

  async *sendMessageStream(
    request: PartListUnion,
    signal: AbortSignal,
    promptId: string,
    turns: number = MAX_TURNS,
    isInvalidStreamRetry: boolean = false,
  ): AsyncGenerator<ServerGeminiStreamEvent, Turn> {
    this.ensurePrompt(promptId);
    const currentTurnCount = this.incrementSessionTurnCount();
    if (
      this.config.getMaxSessionTurns() > 0 &&
      currentTurnCount > this.config.getMaxSessionTurns()
    ) {
      yield { type: GeminiEventType.MaxSessionTurns };
      return new Turn(this.getChat(), promptId);
    }

    const boundedTurns = Math.min(turns, MAX_TURNS);
    if (!boundedTurns) {
      return new Turn(this.getChat(), promptId);
    }

    const modelForLimitCheck = this.getEffectiveModelForCurrentTurn();
    const estimatedRequestTokenCount = Math.floor(
      JSON.stringify(request).length / 4,
    );
    const remainingTokenCount =
      tokenLimit(modelForLimitCheck) - this.getChat().getLastPromptTokenCount();

    if (estimatedRequestTokenCount > remainingTokenCount * 0.95) {
      yield {
        type: GeminiEventType.ContextWindowWillOverflow,
        value: { estimatedRequestTokenCount, remainingTokenCount },
      };
      return new Turn(this.getChat(), promptId);
    }

    const compressed = await this.tryCompressChat(promptId, false);
    if (compressed.compressionStatus === CompressionStatus.COMPRESSED) {
      yield { type: GeminiEventType.ChatCompressed, value: compressed };
    }

    const history = this.getHistory();
    const lastMessage = history.at(-1);
    const hasPendingToolCall = Boolean(
      lastMessage &&
        lastMessage.role === 'model' &&
        lastMessage.parts?.some((part) => 'functionCall' in part),
    );

    if (this.config.getIdeMode() && !hasPendingToolCall) {
      const { contextParts, newIdeContext } = this.getIdeContextUpdate(
        this.forceFullIdeContext || history.length === 0,
      );
      if (contextParts.length > 0) {
        this.getChat().addHistory({
          role: 'user',
          parts: [{ text: contextParts.join('\n') }],
        });
      }
      this.markIdeContextSent(newIdeContext);
    }

    const turn = new Turn(this.getChat(), promptId);
    const controller = new AbortController();
    const linkedSignal = AbortSignal.any([signal, controller.signal]);

    const loopDetected = await this.loopDetector.turnStarted(signal);
    if (loopDetected) {
      yield { type: GeminiEventType.LoopDetected };
      return turn;
    }

    const routingContext: RoutingContext = {
      history: this.getChat().getHistory(/* curated= */ true),
      request,
      signal,
    };

    let modelToUse: string;
    if (this.currentSequenceModel) {
      modelToUse = this.currentSequenceModel;
    } else {
      const router = await this.config.getModelRouterService();
      const decision = await router.route(routingContext);
      modelToUse = decision.model;
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
        if (this.config.getContinueOnFailedApiCall()) {
          if (isInvalidStreamRetry) {
            logContentRetryFailure(
              this.config,
              new ContentRetryFailureEvent(
                4,
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
            promptId,
            boundedTurns - 1,
            true,
          );
          return turn;
        }
      }

      if (event.type === GeminiEventType.Error) {
        return turn;
      }
    }

    if (!turn.pendingToolCalls.length && signal && !signal.aborted) {
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
        promptId,
      );
      logNextSpeakerCheck(
        this.config,
        new NextSpeakerCheckEvent(
          promptId,
          turn.finishReason?.toString() || '',
          nextSpeakerCheck?.next_speaker || '',
        ),
      );
      if (nextSpeakerCheck?.next_speaker === 'model') {
        const nextRequest = [{ text: 'Please continue.' }];
        yield* this.sendMessageStream(
          nextRequest,
          signal,
          promptId,
          boundedTurns - 1,
          false,
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
    let currentAttemptModel: string = model;
    const configToUse: GenerateContentConfig = {
      ...this.baseGenerateConfig,
      ...generationConfig,
    };

    try {
      const userMemory = this.config.getUserMemory();
      const systemInstruction = getCoreSystemPrompt(this.config, userMemory);
      const requestConfig: GenerateContentConfig = {
        abortSignal,
        ...configToUse,
        systemInstruction,
      };

      const apiCall = () => {
        const modelToUse = this.config.isInFallbackMode()
          ? DEFAULT_GEMINI_FLASH_MODEL
          : model;
        currentAttemptModel = modelToUse;

        return this.getContentGeneratorOrFail().generateContent(
          {
            model: modelToUse,
            config: requestConfig,
            contents,
          },
          this.lastPromptId,
        );
      };

      const onPersistent429Callback = async (
        authType?: string,
        error?: unknown,
      ) =>
        await handleFallback(this.config, currentAttemptModel, authType, error);

      return await retryWithBackoff(apiCall, {
        onPersistent429: onPersistent429Callback,
        authType: this.config.getContentGeneratorConfig()?.authType,
      });
    } catch (error: unknown) {
      if (abortSignal.aborted) {
        throw error;
      }

      await reportError(
        error,
        `Error generating content via API with model ${currentAttemptModel}.`,
        {
          requestContents: contents,
          requestConfig: configToUse,
        },
        'generateContent-api',
      );
      throw new Error(
        `Failed to generate content with model ${currentAttemptModel}: ${getErrorMessage(
          error,
        )}`,
      );
    }
  }

  handlePromptChange(promptId: string): void {
    if (this.lastPromptId !== promptId) {
      this.loopDetector.reset(promptId);
      this.setLastPromptId(promptId);
      this.setCurrentSequenceModel(null);
    }
  }

  private updateTelemetryTokenCount() {
    if (this.chat) {
      uiTelemetryService.setLastPromptTokenCount(
        this.chat.getLastPromptTokenCount(),
      );
    }
  }

  private getContentGeneratorOrFail(): ContentGenerator {
    const generator = this.config.getContentGenerator();
    if (!generator) {
      throw new Error('Content generator not initialized');
    }
    return generator;
  }

  private async startChat(
    extraHistory?: Content[],
    resumedSessionData?: ResumedSessionData,
  ): Promise<GeminiChat> {
    this.forceFullIdeContextUpdate();
    this.clearCompressionFailure();

    const toolRegistry = this.config.getToolRegistry();
    const toolDeclarations = toolRegistry.getFunctionDeclarations();
    const tools: Tool[] = [{ functionDeclarations: toolDeclarations }];

    const history = await getInitialChatHistory(this.config, extraHistory);

    try {
      const userMemory = this.config.getUserMemory();
      const systemInstruction = getCoreSystemPrompt(this.config, userMemory);
      const model = this.config.getModel();
      const config: GenerateContentConfig = { ...this.baseGenerateConfig };

      if (isThinkingSupported(model)) {
        config.thinkingConfig = {
          includeThoughts: true,
          thinkingBudget: DEFAULT_THINKING_MODE,
        };
      }

      return new GeminiChat(
        this.config,
        {
          systemInstruction,
          ...config,
          tools,
        },
        history,
        resumedSessionData,
      );
    } catch (error) {
      await reportError(
        error,
        'Error initializing Gemini chat session.',
        history,
        'startChat',
      );
      throw new Error(`Failed to initialize chat: ${getErrorMessage(error)}`);
    }
  }

  private ensurePrompt(promptId: string): void {
    this.handlePromptChange(promptId);
  }

  private getIdeContextUpdate(forceFullContext: boolean): ContextUpdate {
    const currentIdeContext = ideContextStore.get();
    if (!currentIdeContext) {
      return { contextParts: [], newIdeContext: undefined };
    }

    if (forceFullContext || !this.lastSentIdeContext) {
      return buildFullIdeContext(currentIdeContext, this.config.getDebugMode());
    }

    return buildDeltaIdeContext(
      this.lastSentIdeContext,
      currentIdeContext,
      this.config.getDebugMode(),
    );
  }

  private getEffectiveModelForCurrentTurn(): string {
    if (this.currentSequenceModel) {
      return this.currentSequenceModel;
    }

    const configModel = this.config.getModel();
    const model =
      configModel === DEFAULT_GEMINI_MODEL_AUTO
        ? DEFAULT_GEMINI_MODEL
        : configModel;
    return getEffectiveModel(this.config.isInFallbackMode(), model);
  }
}

function buildFullIdeContext(
  currentIdeContext: IdeContext,
  debugMode: boolean,
): ContextUpdate {
  const openFiles = currentIdeContext.workspaceState?.openFiles || [];
  const activeFile = openFiles.find((file) => file.isActive);
  const otherOpenFiles = openFiles
    .filter((file) => !file.isActive)
    .map((file) => file.path);

  const contextData: Record<string, unknown> = {};

  if (activeFile) {
    contextData['activeFile'] = {
      path: activeFile.path,
      cursor: activeFile.cursor
        ? {
            line: activeFile.cursor.line,
            character: activeFile.cursor.character,
          }
        : undefined,
      selectedText: activeFile.selectedText || undefined,
    };
  }

  if (otherOpenFiles.length > 0) {
    contextData['otherOpenFiles'] = otherOpenFiles;
  }

  if (Object.keys(contextData).length === 0) {
    return { contextParts: [], newIdeContext: currentIdeContext };
  }

  const jsonString = JSON.stringify(contextData, null, 2);
  const contextParts = [
    "Here is the user's editor context as a JSON object. This is for your information only.",
    '```json',
    jsonString,
    '```',
  ];

  if (debugMode) {
    debugLogger.log(contextParts.join('\n'));
  }

  return {
    contextParts,
    newIdeContext: currentIdeContext,
  };
}

function buildDeltaIdeContext(
  previousIdeContext: IdeContext,
  currentIdeContext: IdeContext,
  debugMode: boolean,
): ContextUpdate {
  const delta: Record<string, unknown> = {};
  const changes: Record<string, unknown> = {};

  const lastFiles = new Map(
    (previousIdeContext.workspaceState?.openFiles || []).map((file: File) => [
      file.path,
      file,
    ]),
  );
  const currentFiles = new Map(
    (currentIdeContext.workspaceState?.openFiles || []).map((file: File) => [
      file.path,
      file,
    ]),
  );

  const openedFiles: string[] = [];
  for (const [path] of currentFiles.entries()) {
    if (!lastFiles.has(path)) {
      openedFiles.push(path);
    }
  }
  if (openedFiles.length > 0) {
    changes['filesOpened'] = openedFiles;
  }

  const closedFiles: string[] = [];
  for (const [path] of lastFiles.entries()) {
    if (!currentFiles.has(path)) {
      closedFiles.push(path);
    }
  }
  if (closedFiles.length > 0) {
    changes['filesClosed'] = closedFiles;
  }

  const lastActiveFile =
    previousIdeContext.workspaceState?.openFiles?.find((file: File) =>
      Boolean(file.isActive),
    );
  const currentActiveFile =
    currentIdeContext.workspaceState?.openFiles?.find((file: File) =>
      Boolean(file.isActive),
    );

  if (currentActiveFile) {
    if (!lastActiveFile || lastActiveFile.path !== currentActiveFile.path) {
      changes['activeFileChanged'] = {
        path: currentActiveFile.path,
        cursor: currentActiveFile.cursor
          ? {
              line: currentActiveFile.cursor.line,
              character: currentActiveFile.cursor.character,
            }
          : undefined,
        selectedText: currentActiveFile.selectedText || undefined,
      };
    } else {
      const lastCursor = lastActiveFile.cursor;
      const currentCursor = currentActiveFile.cursor;
      if (
        currentCursor &&
        (!lastCursor ||
          lastCursor.line !== currentCursor.line ||
          lastCursor.character !== currentCursor.character)
      ) {
        changes['cursorMoved'] = {
          path: currentActiveFile.path,
          cursor: {
            line: currentCursor.line,
            character: currentCursor.character,
          },
        };
      }

      const lastSelectedText = lastActiveFile.selectedText || '';
      const currentSelectedText = currentActiveFile.selectedText || '';
      if (lastSelectedText !== currentSelectedText) {
        changes['selectionChanged'] = {
          path: currentActiveFile.path,
          selectedText: currentSelectedText,
        };
      }
    }
  } else if (lastActiveFile) {
    changes['activeFileChanged'] = {
      path: null,
      previousPath: lastActiveFile.path,
    };
  }

  if (Object.keys(changes).length === 0) {
    return { contextParts: [], newIdeContext: currentIdeContext };
  }

  delta['changes'] = changes;
  const jsonString = JSON.stringify(delta, null, 2);
  const contextParts = [
    "Here is a summary of changes in the user's editor context, in JSON format. This is for your information only.",
    '```json',
    jsonString,
    '```',
  ];

  if (debugMode) {
    debugLogger.log(contextParts.join('\n'));
  }

  return {
    contextParts,
    newIdeContext: currentIdeContext,
  };
}

