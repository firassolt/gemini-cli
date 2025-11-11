/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { AsyncGenerator } from 'node:stream';
import type { GenerateContentResponse } from '@google/genai';

import type { Config } from '../config/config.js';
import {
  logContentRetry,
  logContentRetryFailure,
  logStreamRetryAttempt,
  logStreamRetryFailure,
} from '../telemetry/loggers.js';
import {
  ContentRetryEvent,
  ContentRetryFailureEvent,
  StreamRetryAttemptEvent,
  StreamRetryFailureEvent,
} from '../telemetry/types.js';
import {
  InvalidStreamError,
  type InvalidStreamErrorContext,
  type InvalidStreamErrorType,
  type InvalidStreamErrorCategory,
} from './invalidStreamError.js';

export interface RetryControllerOptions {
  maxAttempts: number;
  initialDelayMs: number;
}

export interface RetryEventPayload {
  attempt: number;
  maxAttempts: number;
  delayMs: number;
  errorType: InvalidStreamErrorType;
  errorCategory: InvalidStreamErrorCategory;
  context?: InvalidStreamErrorContext;
}

interface RetryDecision {
  shouldRetry: boolean;
  payload?: RetryEventPayload;
  delayMs?: number;
}

function toTelemetryCategory(
  category: InvalidStreamErrorCategory,
): 'safety_block' | 'malformed_content' {
  return category === 'SAFETY_BLOCK' ? 'safety_block' : 'malformed_content';
}

export type RetryControllerYield =
  | { kind: 'chunk'; value: GenerateContentResponse }
  | { kind: 'retry'; value: RetryEventPayload };

export class RetryController {
  constructor(
    private readonly config: Config,
    private readonly model: string,
    private readonly promptId: string,
    private readonly options: RetryControllerOptions,
  ) {}

  async *run(
    attemptFactory: (
      attempt: number,
    ) => Promise<AsyncGenerator<GenerateContentResponse>>,
  ): AsyncGenerator<RetryControllerYield> {
    let lastError: unknown = new Error('Request failed after all retries.');

    for (let attempt = 0; attempt < this.options.maxAttempts; attempt++) {
      try {
        const stream = await attemptFactory(attempt);
        for await (const chunk of stream) {
          yield { kind: 'chunk', value: chunk } satisfies RetryControllerYield;
        }
        lastError = null;
        break;
      } catch (error: unknown) {
        lastError = error;
        const decision = await this.handleFailure(attempt, error);
        if (decision.payload && decision.shouldRetry) {
          yield { kind: 'retry', value: decision.payload } satisfies RetryControllerYield;
        }
        if (!decision.shouldRetry) {
          break;
        }
        if (decision.delayMs && decision.delayMs > 0) {
          await new Promise((resolve) => setTimeout(resolve, decision.delayMs));
        }
      }
    }

    if (lastError) {
      throw lastError;
    }
  }

  private async handleFailure(
    attempt: number,
    error: unknown,
  ): Promise<RetryDecision> {
    if (!(error instanceof InvalidStreamError)) {
      return { shouldRetry: false };
    }

    const delayMs = this.options.initialDelayMs * (attempt + 1);
    const payload: RetryEventPayload = {
      attempt,
      maxAttempts: this.options.maxAttempts,
      delayMs,
      errorType: error.type,
      errorCategory: error.category,
      context: error.context
        ? structuredClone(error.context)
        : undefined,
    };

    if (error.category === 'SAFETY_BLOCK') {
      this.logRetryFailure(payload);
      return { shouldRetry: false, payload };
    }

    if (attempt < this.options.maxAttempts - 1) {
      this.logRetryAttempt(payload);
      return { shouldRetry: true, payload, delayMs };
    }

    this.logRetryFailure(payload);
    return { shouldRetry: false, payload };
  }

  private logRetryAttempt(payload: RetryEventPayload): void {
    logContentRetry(
      this.config,
      new ContentRetryEvent(
        payload.attempt,
        payload.errorType,
        payload.delayMs,
        this.model,
      ),
    );

    logStreamRetryAttempt(
      this.config,
      new StreamRetryAttemptEvent(
        payload.attempt,
        payload.maxAttempts,
        toTelemetryCategory(payload.errorCategory),
        payload.errorType,
        this.model,
        this.promptId,
        Boolean(payload.context?.partialResponseParts?.length),
      ),
    );
  }

  private logRetryFailure(payload: RetryEventPayload): void {
    logContentRetryFailure(
      this.config,
      new ContentRetryFailureEvent(
        payload.maxAttempts,
        payload.errorType,
        this.model,
      ),
    );

    logStreamRetryFailure(
      this.config,
      new StreamRetryFailureEvent(
        payload.maxAttempts,
        toTelemetryCategory(payload.errorCategory),
        payload.errorType,
        this.model,
        this.promptId,
        Boolean(payload.context?.partialResponseParts?.length),
      ),
    );
  }
}
