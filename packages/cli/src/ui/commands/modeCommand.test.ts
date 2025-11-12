/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it, vi } from 'vitest';
import { modeCommand } from './modeCommand.js';
import { SessionMode } from '@google/gemini-cli-core';
import { createMockCommandContext } from '../../test-utils/mockCommandContext.js';

describe('modeCommand', () => {
  it('reports current mode when no arguments provided', () => {
    const context = createMockCommandContext({
      session: { mode: SessionMode.PLAN },
    });

    const result = modeCommand.action?.(context, '') as {
      type: string;
      content: string;
    };

    expect(result.type).toBe('message');
    expect(result.content).toContain('Plan');
  });

  it('switches to build mode', () => {
    const setMode = vi.fn();
    const context = createMockCommandContext({
      session: {
        mode: SessionMode.PLAN,
        setMode,
      },
    });

    const result = modeCommand.action?.(context, 'build');

    expect(setMode).toHaveBeenCalledWith(SessionMode.BUILD);
    expect(result).toEqual({
      type: 'message',
      messageType: 'info',
      content:
        'Build mode enabled. Modifying tools require explicit approval before execution.',
    });
  });
});
