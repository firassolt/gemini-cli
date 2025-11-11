/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import path from 'node:path';
import { parse as parseYaml } from 'yaml';
import picomatch from 'picomatch';
import { z } from 'zod';
import type { AnyToolInvocation, ToolLocation } from '../tools/tools.js';
import type { StructuredError } from '../core/turn.js';

const ResponseRequirementSchema = z
  .object({
    id: z.string(),
    description: z.string(),
    example: z.string().optional(),
  })
  .strict();

const FileScopeSchema = z
  .object({
    id: z.string().optional(),
    name: z.string().optional(),
    description: z.string().optional(),
    include: z.array(z.string()).nonempty(),
    exclude: z.array(z.string()).optional(),
  })
  .strict();

const PolicyRulesSchema = z
  .object({
    allowedTools: z.array(z.string()).optional(),
    fileScopes: z.array(FileScopeSchema).optional(),
    responseRequirements: z.array(ResponseRequirementSchema).optional(),
  })
  .strict();

const PolicyConfigSchema = z
  .object({
    path: z.string().optional(),
    rules: PolicyRulesSchema.optional(),
  })
  .strict();

export type DeclarativeResponseRequirement = z.infer<
  typeof ResponseRequirementSchema
>;
export type DeclarativeFileScope = z.infer<typeof FileScopeSchema>;
export type DeclarativePolicyRules = z.infer<typeof PolicyRulesSchema>;
export type AgentPolicyConfig = z.infer<typeof PolicyConfigSchema>;

export interface DeclarativePolicyLoadOptions {
  projectRoot: string;
  readTextFile: (absolutePath: string) => Promise<string>;
}

export type DeclarativePolicySource =
  | AgentPolicyConfig
  | DeclarativePolicyRules
  | undefined;

export interface DeclarativePolicyViolation {
  code: 'TOOL_NOT_ALLOWED' | 'FILE_SCOPE_VIOLATION';
  message: string;
  toolName: string;
  path?: string;
}

function normalizeToPosix(relativePath: string): string {
  return relativePath.split(path.sep).join('/');
}

function parsePolicyDocument(documentText: string): unknown {
  const trimmed = documentText.trim();
  if (!trimmed) {
    return {};
  }

  try {
    return JSON.parse(trimmed);
  } catch {
    return parseYaml(trimmed);
  }
}

function mergePolicyRules(
  base: DeclarativePolicyRules,
  override?: DeclarativePolicyRules,
): DeclarativePolicyRules {
  if (!override) {
    return base;
  }

  return {
    allowedTools: override.allowedTools ?? base.allowedTools,
    fileScopes: override.fileScopes ?? base.fileScopes,
    responseRequirements:
      override.responseRequirements ?? base.responseRequirements,
  };
}

class FileScopeMatcher {
  private readonly includeMatchers: Array<(value: string) => boolean>;
  private readonly excludeMatchers: Array<(value: string) => boolean>;
  private readonly label: string;

  constructor(
    readonly rule: DeclarativeFileScope,
    private readonly projectRoot: string,
  ) {
    this.includeMatchers = rule.include.map((pattern) =>
      picomatch(pattern, { dot: true }),
    );
    this.excludeMatchers = (rule.exclude ?? []).map((pattern) =>
      picomatch(pattern, { dot: true }),
    );
    this.label =
      rule.name ?? rule.id ?? `scope_${Math.random().toString(16).slice(2, 8)}`;
  }

  matches(absolutePath: string): boolean {
    const relative = path.relative(this.projectRoot, absolutePath);
    if (!relative || relative.startsWith('..')) {
      return false;
    }
    const normalized = normalizeToPosix(relative);
    const included = this.includeMatchers.some((matcher) => matcher(normalized));
    if (!included) {
      return false;
    }
    const excluded = this.excludeMatchers.some((matcher) => matcher(normalized));
    return !excluded;
  }

  describe(): string {
    const includeList = this.rule.include.join(', ');
    const excludeList = (this.rule.exclude ?? []).join(', ');
    const description = this.rule.description
      ? `${this.rule.description} `
      : '';
    const excludeText = excludeList ? ` Excluding: ${excludeList}.` : '';
    return `${this.label}: ${description}Including: ${includeList}.${excludeText}`.trim();
  }

  get id(): string {
    return this.rule.id ?? this.label;
  }
}

export class DeclarativePolicyViolationError extends Error {
  readonly violation: DeclarativePolicyViolation;
  readonly structuredError: StructuredError;

  constructor(violation: DeclarativePolicyViolation) {
    super(violation.message);
    this.name = 'DeclarativePolicyViolationError';
    this.violation = violation;
    this.structuredError = {
      message: violation.message,
      status: 412,
    };
    Error.captureStackTrace?.(this, DeclarativePolicyViolationError);
  }
}

export class DeclarativeTaskPolicy {
  private readonly allowedTools?: Set<string>;
  private readonly fileScopes: FileScopeMatcher[];
  private readonly responseRequirements: DeclarativeResponseRequirement[];

  constructor(
    rules: DeclarativePolicyRules,
    private readonly projectRoot: string,
  ) {
    this.allowedTools = rules.allowedTools
      ? new Set(rules.allowedTools)
      : undefined;
    this.fileScopes = (rules.fileScopes ?? []).map(
      (scope) => new FileScopeMatcher(scope, projectRoot),
    );
    this.responseRequirements = rules.responseRequirements ?? [];
  }

  describeForPrompt(): string {
    const sections: string[] = [];
    sections.push('You must comply with the declarative policy rules below.');

    if (this.allowedTools) {
      sections.push(
        `- Allowed tools: ${[...this.allowedTools].sort().join(', ') || 'None'}.`,
      );
    }

    if (this.fileScopes.length > 0) {
      sections.push('- File scopes:');
      for (const scope of this.fileScopes) {
        sections.push(`  • ${scope.describe()}`);
      }
    }

    if (this.responseRequirements.length > 0) {
      sections.push('- Response requirements:');
      for (const requirement of this.responseRequirements) {
        sections.push(`  • ${requirement.id}: ${requirement.description}`);
      }
    }

    return sections.join('\n');
  }

  validateToolCall(
    toolName: string,
    invocation: AnyToolInvocation,
  ): DeclarativePolicyViolation | null {
    if (this.allowedTools && !this.allowedTools.has(toolName)) {
      const allowedList = [...this.allowedTools].sort().join(', ') || 'none';
      return {
        code: 'TOOL_NOT_ALLOWED',
        message: `Tool "${toolName}" is blocked by declarative policy. Allowed tools: ${allowedList}.`,
        toolName,
      };
    }

    if (this.fileScopes.length === 0) {
      return null;
    }

    const locations: ToolLocation[] = invocation.toolLocations?.() ?? [];
    for (const location of locations) {
      const matches = this.fileScopes.find((scope) =>
        scope.matches(location.path),
      );
      if (!matches) {
        return {
          code: 'FILE_SCOPE_VIOLATION',
          message: `Tool "${toolName}" attempted to access "${path.relative(
            this.projectRoot,
            location.path,
          )}" which is outside of the allowed policy scopes.`,
          toolName,
          path: location.path,
        };
      }
    }

    return null;
  }

  getResponseRequirements(): readonly DeclarativeResponseRequirement[] {
    return this.responseRequirements;
  }
}

function normalizePolicySource(
  source: DeclarativePolicySource,
): AgentPolicyConfig | undefined {
  if (!source) {
    return undefined;
  }

  if ('allowedTools' in source || 'fileScopes' in source || 'responseRequirements' in source) {
    return { rules: source as DeclarativePolicyRules };
  }

  return source as AgentPolicyConfig;
}

export async function loadDeclarativePolicy(
  source: DeclarativePolicySource,
  options: DeclarativePolicyLoadOptions,
): Promise<DeclarativeTaskPolicy | undefined> {
  const normalized = normalizePolicySource(source);
  if (!normalized) {
    return undefined;
  }

  let mergedRules: DeclarativePolicyRules | undefined = normalized.rules;

  if (normalized.path) {
    const absolutePath = path.isAbsolute(normalized.path)
      ? normalized.path
      : path.resolve(options.projectRoot, normalized.path);

    let fileContent: string;
    try {
      fileContent = await options.readTextFile(absolutePath);
    } catch (error) {
      throw new Error(
        `Failed to read declarative policy file at "${absolutePath}": ${String(
          error,
        )}`,
      );
    }

    const parsed = parsePolicyDocument(fileContent);
    const parsedRulesResult = PolicyRulesSchema.safeParse(parsed);
    if (!parsedRulesResult.success) {
      throw new Error(
        `Invalid declarative policy file at "${absolutePath}": ${parsedRulesResult.error.message}`,
      );
    }

    mergedRules = mergePolicyRules(parsedRulesResult.data, mergedRules);
  }

  if (!mergedRules) {
    return undefined;
  }

  const rulesResult = PolicyRulesSchema.safeParse(mergedRules);
  if (!rulesResult.success) {
    throw new Error(
      `Invalid declarative policy configuration: ${rulesResult.error.message}`,
    );
  }

  return new DeclarativeTaskPolicy(rulesResult.data, options.projectRoot);
}
