/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  Candidate,
  GenerateContentResponsePromptFeedback,
  Part,
} from '@google/genai';

export type InvalidStreamErrorType =
  | 'NO_FINISH_REASON'
  | 'NO_RESPONSE_TEXT'
  | 'SAFETY_BLOCKED';

export type InvalidStreamErrorCategory =
  | 'SAFETY_BLOCK'
  | 'MALFORMED_CONTENT';

export interface InvalidStreamErrorContext {
  partialResponseParts?: Part[];
  finishReason?: string;
  promptFeedback?: GenerateContentResponsePromptFeedback;
  safetyRatings?: Candidate['safetyRatings'];
}

export class InvalidStreamError extends Error {
  readonly type: InvalidStreamErrorType;
  readonly category: InvalidStreamErrorCategory;
  readonly context?: InvalidStreamErrorContext;

  constructor(
    message: string,
    type: InvalidStreamErrorType,
    context?: InvalidStreamErrorContext,
  ) {
    super(message);
    this.name = 'InvalidStreamError';
    this.type = type;
    this.category =
      type === 'SAFETY_BLOCKED' ? 'SAFETY_BLOCK' : 'MALFORMED_CONTENT';
    this.context = context;
  }
}
