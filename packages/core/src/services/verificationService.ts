/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { PartListUnion } from '@google/genai';
import type { Config } from '../config/config.js';
import { partListUnionToString } from '../core/geminiRequest.js';
import { WEB_SEARCH_TOOL_NAME } from '../tools/tool-names.js';
import type { ToolInvocation } from '../tools/tools.js';
import type { WebSearchToolResult } from '../tools/web-search.js';
import { debugLogger } from '../utils/debugLogger.js';

export type VerificationStatus = 'grounded' | 'uncertain' | 'skipped';

export interface VerificationAssertion {
  assertion: string;
  grounded: boolean;
  sources: Array<{ title?: string; uri?: string }>;
  snippet?: string;
  note?: string;
}

export interface VerificationResult {
  required: boolean;
  status: VerificationStatus;
  reason?: string;
  query?: string;
  summaryText: string;
  assertions: VerificationAssertion[];
}

export interface VerificationInput {
  finalText: string;
  promptId: string;
  model: string;
  reason?: string;
  signal: AbortSignal;
}

const MAX_QUERY_LENGTH = 256;
const MAX_SNIPPET_LENGTH = 400;

function toString(content: PartListUnion): string {
  if (typeof content === 'string') {
    return content;
  }
  return partListUnionToString(content);
}

function truncate(value: string, maxLength: number): string {
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, maxLength - 1)}…`;
}

function formatSources(
  sources: Array<{ title?: string; uri?: string }>,
): string {
  if (!sources.length) {
    return '';
  }
  const lines = sources.map((source, index) => {
    const title = source.title ?? 'Untitled';
    const uri = source.uri ?? 'No URI available';
    return `  ${index + 1}. ${title} (${uri})`;
  });
  return ['Sources:', ...lines].join('\n');
}

function quoteSnippet(snippet: string | undefined): string {
  if (!snippet) {
    return '';
  }
  const normalised = snippet.trim().replace(/\s+/g, ' ');
  if (!normalised) {
    return '';
  }
  return ['Snippet:', `> ${normalised}`].join('\n');
}

export class VerificationService {
  constructor(private readonly config: Config) {}

  async verify(input: VerificationInput): Promise<VerificationResult> {
    const required = true;
    const normalizedText = input.finalText.trim();
    if (!normalizedText) {
      const summary =
        'Verification result:\n- ⚠️ Unable to verify because no response text was produced.';
      return {
        required,
        status: 'uncertain',
        reason: 'empty_response',
        summaryText: summary,
        assertions: [
          {
            assertion: '',
            grounded: false,
            sources: [],
            note: 'No response text available to verify.',
          },
        ],
      };
    }

    const query = this.buildQuery(normalizedText, input.reason);

    const registry = this.config.getToolRegistry();
    const tool = registry.getTool(WEB_SEARCH_TOOL_NAME);
    if (!tool) {
      const summary =
        'Verification result:\n- ⚠️ Web search tool is unavailable. Treat this answer as unverified.';
      return {
        required,
        status: 'uncertain',
        reason: 'missing_web_search_tool',
        query,
        summaryText: summary,
        assertions: [
          {
            assertion: truncate(normalizedText, MAX_SNIPPET_LENGTH),
            grounded: false,
            sources: [],
            note: 'Web search tool not registered in tool registry.',
          },
        ],
      };
    }

    let invocation: ToolInvocation<{ query: string }, WebSearchToolResult>;
    try {
      invocation = tool.build({ query });
    } catch (error) {
      const message = `Unable to prepare verification search: ${String(error)}`;
      debugLogger.error('[VerificationService] build failed', error);
      return {
        required,
        status: 'uncertain',
        reason: 'web_search_build_failed',
        query,
        summaryText: `Verification result:\n- ⚠️ ${message}`,
        assertions: [
          {
            assertion: truncate(normalizedText, MAX_SNIPPET_LENGTH),
            grounded: false,
            sources: [],
            note: message,
          },
        ],
      };
    }

    const controller = new AbortController();
    const combinedSignal = AbortSignal.any([input.signal, controller.signal]);
    try {
      const result = await invocation.execute(combinedSignal);
      const snippet = truncate(toString(result.llmContent).trim(), MAX_SNIPPET_LENGTH);
      const sources = (result.sources ?? []).map((source) => ({
        title: source.web?.title ?? undefined,
        uri: source.web?.uri ?? undefined,
      }));

      if (sources.length === 0) {
        const summaryLines = [
          'Verification result:',
          '- ⚠️ No supporting sources were returned. Treat this answer as unverified.',
        ];
        return {
          required,
          status: 'uncertain',
          reason: 'no_sources',
          query,
          summaryText: summaryLines.join('\n'),
          assertions: [
            {
              assertion: truncate(normalizedText, MAX_SNIPPET_LENGTH),
              grounded: false,
              sources: [],
              snippet,
              note: 'Web search returned no grounding sources.',
            },
          ],
        };
      }

      const summaryLines = [
        'Verification result:',
        `- ✅ Grounded with ${sources.length} supporting source${sources.length === 1 ? '' : 's'}.`,
      ];
      const snippetSection = quoteSnippet(snippet);
      if (snippetSection) {
        summaryLines.push('', snippetSection);
      }
      const sourcesSection = formatSources(sources);
      if (sourcesSection) {
        summaryLines.push('', sourcesSection);
      }
      return {
        required,
        status: 'grounded',
        reason: input.reason,
        query,
        summaryText: summaryLines.join('\n'),
        assertions: [
          {
            assertion: truncate(normalizedText, MAX_SNIPPET_LENGTH),
            grounded: true,
            sources,
            snippet,
          },
        ],
      };
    } catch (error) {
      if (combinedSignal.aborted) {
        const summary =
          'Verification result:\n- ⚠️ Verification cancelled before completion. Treat this answer as unverified.';
        return {
          required,
          status: 'uncertain',
          reason: 'aborted',
          query,
          summaryText: summary,
          assertions: [
            {
              assertion: truncate(normalizedText, MAX_SNIPPET_LENGTH),
              grounded: false,
              sources: [],
              note: 'Verification aborted.',
            },
          ],
        };
      }
      const message = `Verification failed: ${String(error)}`;
      debugLogger.error('[VerificationService] execute failed', error);
      return {
        required,
        status: 'uncertain',
        reason: 'web_search_failed',
        query,
        summaryText: `Verification result:\n- ⚠️ ${message}`,
        assertions: [
          {
            assertion: truncate(normalizedText, MAX_SNIPPET_LENGTH),
            grounded: false,
            sources: [],
            note: message,
          },
        ],
      };
    } finally {
      controller.abort();
    }
  }

  private buildQuery(response: string, reason?: string): string {
    const base = reason ? `${reason}: ${response}` : response;
    const normalised = base.replace(/\s+/g, ' ').trim();
    if (normalised.length <= MAX_QUERY_LENGTH) {
      return normalised;
    }
    return normalised.slice(0, MAX_QUERY_LENGTH - 1) + '…';
  }
}
