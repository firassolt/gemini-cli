/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi } from 'vitest';
import {
  DeclarativeTaskPolicy,
  loadDeclarativePolicy,
} from './declarativePolicy.js';
import { MockTool } from '../test-utils/mock-tool.js';

const EMPTY_SCHEMA = { type: 'object', properties: {} } as const;

describe('DeclarativeTaskPolicy', () => {
  it('denies tool calls that are not explicitly allowed', () => {
    const policy = new DeclarativeTaskPolicy(
      { allowedTools: ['read_file'] },
      '/workspace',
    );
    const tool = new MockTool({ name: 'write_file', params: EMPTY_SCHEMA });
    const invocation = tool.build({});

    const violation = policy.validateToolCall('write_file', invocation);

    expect(violation?.code).toBe('TOOL_NOT_ALLOWED');
    expect(violation?.message).toContain('read_file');
  });

  it('denies file operations outside of declared scopes', () => {
    const policy = new DeclarativeTaskPolicy(
      { fileScopes: [{ include: ['src/**'] }] },
      '/workspace',
    );
    const tool = new MockTool({ name: 'write_file', params: EMPTY_SCHEMA });
    tool.locations = [{ path: '/workspace/README.md' }];
    const invocation = tool.build({});

    const violation = policy.validateToolCall('write_file', invocation);

    expect(violation?.code).toBe('FILE_SCOPE_VIOLATION');
    expect(violation?.message).toContain('README.md');
  });

  it('allows file operations when paths satisfy the scope', () => {
    const policy = new DeclarativeTaskPolicy(
      { fileScopes: [{ include: ['src/**'] }] },
      '/workspace',
    );
    const tool = new MockTool({ name: 'write_file', params: EMPTY_SCHEMA });
    tool.locations = [{ path: '/workspace/src/index.ts' }];
    const invocation = tool.build({});

    const violation = policy.validateToolCall('write_file', invocation);

    expect(violation).toBeNull();
  });

  it('produces a descriptive prompt section', () => {
    const policy = new DeclarativeTaskPolicy(
      {
        allowedTools: ['read_file', 'write_file'],
        fileScopes: [
          {
            name: 'Sources',
            include: ['src/**'],
            exclude: ['src/vendor/**'],
            description: 'Primary application code.',
          },
        ],
        responseRequirements: [
          { id: 'summary', description: 'Summarize your changes.' },
        ],
      },
      '/workspace',
    );

    const description = policy.describeForPrompt();

    expect(description).toContain('Allowed tools');
    expect(description).toContain('Sources');
    expect(description).toContain('summary');
  });
});

describe('loadDeclarativePolicy', () => {
  const readTextFile = vi.fn<Parameters<(path: string) => Promise<string>>, string>();

  beforeEach(() => {
    readTextFile.mockReset();
  });

  it('loads policy rules from YAML files', async () => {
    readTextFile.mockResolvedValueOnce('allowedTools:\n  - write_file');

    const policy = await loadDeclarativePolicy(
      { path: 'policy.yaml' },
      {
        projectRoot: '/repo',
        readTextFile,
      },
    );

    expect(readTextFile).toHaveBeenCalledWith('/repo/policy.yaml');
    expect(policy).toBeInstanceOf(DeclarativeTaskPolicy);

    const tool = new MockTool({ name: 'write_file', params: EMPTY_SCHEMA });
    const invocation = tool.build({});
    expect(policy?.validateToolCall('write_file', invocation)).toBeNull();
  });

  it('returns undefined when no policy source is provided', async () => {
    const policy = await loadDeclarativePolicy(undefined, {
      projectRoot: '/repo',
      readTextFile,
    });

    expect(policy).toBeUndefined();
    expect(readTextFile).not.toHaveBeenCalled();
  });
});
