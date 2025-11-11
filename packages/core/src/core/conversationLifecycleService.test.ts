/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ConversationLifecycleService } from './conversationLifecycleService.js';
import type { Config } from '../config/config.js';
import type { LoopDetectionService } from '../services/loopDetectionService.js';
import type { ChatCompressionService } from '../services/chatCompressionService.js';
import type { ChatRecordingService } from '../services/chatRecordingService.js';
import type { RoutingDecision } from '../routing/routingStrategy.js';
import type { Turn } from './turn.js';
import type { GeminiChat } from './geminiChat.js';
import { VerificationEvent } from '../telemetry/types.js';
import * as telemetryLoggers from '../telemetry/loggers.js';

function setChatRecording(
  service: ConversationLifecycleService,
  recording: ChatRecordingService,
): void {
  const chat = {
    getChatRecordingService: () => recording,
  } as unknown as GeminiChat;
  (service as unknown as { chat?: GeminiChat }).chat = chat;
}

describe('ConversationLifecycleService', () => {
  let config: Config;
  let loopDetector: LoopDetectionService;
  let compressionService: ChatCompressionService;
  let service: ConversationLifecycleService;

  beforeEach(() => {
    config = {
      getSessionId: vi.fn().mockReturnValue('session-123'),
      getUsageStatisticsEnabled: vi.fn().mockReturnValue(false),
    } as unknown as Config;
    loopDetector = {} as LoopDetectionService;
    compressionService = {} as ChatCompressionService;
    service = new ConversationLifecycleService(
      config,
      loopDetector,
      compressionService,
    );
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2025-01-01T12:00:00.000Z'));
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('records verification results and logs telemetry', () => {
    const recording = {
      recordGrounding: vi.fn(),
    } as unknown as ChatRecordingService;
    setChatRecording(service, recording);

    const verificationResult = {
      required: true,
      status: 'grounded' as const,
      reason: 'classifier_complexity',
      query: 'search query',
      summaryText: 'summary',
      assertions: [
        {
          assertion: 'Answer',
          grounded: true,
          sources: [],
          snippet: 'Answer',
        },
        {
          assertion: 'Follow-up',
          grounded: false,
          sources: [],
          note: 'uncertain',
        },
      ],
    };

    const turn = {
      getVerificationResult: vi.fn().mockReturnValue(verificationResult),
    } as unknown as Turn;

    const routingDecision: RoutingDecision = {
      model: 'gemini-pro',
      metadata: {
        source: 'classifier',
        latencyMs: 10,
        reasoning: 'complex task',
        requiresVerification: true,
        verificationReason: 'policy_auto_edit',
      },
    };

    const logSpy = vi
      .spyOn(telemetryLoggers, 'logVerification')
      .mockImplementation(() => {});

    (service as unknown as {
      recordVerificationResult(
        turn: Turn,
        promptId: string,
        model: string,
        routingDecision?: RoutingDecision,
      ): void;
    }).recordVerificationResult(turn, 'prompt-123', 'gemini-pro', routingDecision);

    expect(recording.recordGrounding).toHaveBeenCalledWith({
      required: true,
      status: 'grounded',
      reason: 'classifier_complexity',
      query: 'search query',
      assertions: verificationResult.assertions,
    });

    expect(logSpy).toHaveBeenCalledTimes(1);
    const [, event] = logSpy.mock.calls[0];
    expect(event).toBeInstanceOf(VerificationEvent);
    const verificationEvent = event as VerificationEvent;
    expect(verificationEvent.prompt_id).toBe('prompt-123');
    expect(verificationEvent.model).toBe('gemini-pro');
    expect(verificationEvent.required).toBe(true);
    expect(verificationEvent.status).toBe('grounded');
    expect(verificationEvent.grounded_assertions).toBe(1);
    expect(verificationEvent.total_assertions).toBe(2);
    expect(verificationEvent.reason).toBe('classifier_complexity');
  });

  it('falls back to routing reason when verification result omits a reason', () => {
    const recording = {
      recordGrounding: vi.fn(),
    } as unknown as ChatRecordingService;
    setChatRecording(service, recording);

    const verificationResult = {
      required: true,
      status: 'uncertain' as const,
      summaryText: 'summary',
      assertions: [
        {
          assertion: 'Answer',
          grounded: false,
          sources: [],
          note: 'no sources',
        },
      ],
    };

    const turn = {
      getVerificationResult: vi.fn().mockReturnValue(verificationResult),
    } as unknown as Turn;

    const routingDecision: RoutingDecision = {
      model: 'gemini-pro',
      metadata: {
        source: 'classifier',
        latencyMs: 5,
        reasoning: 'policy enforcement',
        requiresVerification: true,
        verificationReason: 'policy_auto_edit',
      },
    };

    const logSpy = vi
      .spyOn(telemetryLoggers, 'logVerification')
      .mockImplementation(() => {});

    (service as unknown as {
      recordVerificationResult(
        turn: Turn,
        promptId: string,
        model: string,
        routingDecision?: RoutingDecision,
      ): void;
    }).recordVerificationResult(turn, 'prompt-456', 'gemini-pro', routingDecision);

    expect(recording.recordGrounding).toHaveBeenCalledWith({
      required: true,
      status: 'uncertain',
      reason: 'policy_auto_edit',
      query: undefined,
      assertions: verificationResult.assertions,
    });

    const [, event] = logSpy.mock.calls[0];
    const verificationEvent = event as VerificationEvent;
    expect(verificationEvent.reason).toBe('policy_auto_edit');
  });
});
