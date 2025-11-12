/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { SessionMode } from '@google/gemini-cli-core';
import type { SlashCommand, MessageActionReturn } from './types.js';
import { CommandKind } from './types.js';

function formatMode(mode: SessionMode): string {
  return mode === SessionMode.BUILD ? 'Build' : 'Plan';
}

export const modeCommand: SlashCommand = {
  name: 'mode',
  description: 'View or change the current session mode (plan or build).',
  kind: CommandKind.BUILT_IN,
  action: (context, rawArgs): MessageActionReturn => {
    const args = rawArgs.trim().toLowerCase();
    const currentMode = context.session.mode;

    if (!args) {
      return {
        type: 'message',
        messageType: 'info',
        content:
          `Current mode: ${formatMode(currentMode)}. ` +
          "Use '/mode plan' or '/mode build' to switch modes.",
      };
    }

    if (args !== 'plan' && args !== 'build') {
      return {
        type: 'message',
        messageType: 'error',
        content: "Invalid mode. Accepted values are 'plan' or 'build'.",
      };
    }

    const targetMode =
      args === 'plan' ? SessionMode.PLAN : SessionMode.BUILD;

    if (targetMode === currentMode) {
      return {
        type: 'message',
        messageType: 'info',
        content: `Already in ${formatMode(currentMode)} mode.`,
      };
    }

    context.session.setMode(targetMode);

    const message =
      targetMode === SessionMode.BUILD
        ? 'Build mode enabled. Modifying tools require explicit approval before execution.'
        : 'Plan mode enabled. Tool access limited to read-only operations.';

    return {
      type: 'message',
      messageType: 'info',
      content: message,
    };
  },
};
