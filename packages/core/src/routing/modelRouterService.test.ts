/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ModelRouterService } from './modelRouterService.js';
import { Config } from '../config/config.js';
import type { BaseLlmClient } from '../core/baseLlmClient.js';
import type { RoutingContext, RoutingDecision } from './routingStrategy.js';
import { DefaultStrategy } from './strategies/defaultStrategy.js';
import { CompositeStrategy } from './strategies/compositeStrategy.js';
import { FallbackStrategy } from './strategies/fallbackStrategy.js';
import { OverrideStrategy } from './strategies/overrideStrategy.js';
import { ClassifierStrategy } from './strategies/classifierStrategy.js';
import { logModelRouting } from '../telemetry/loggers.js';
import { ModelRoutingEvent } from '../telemetry/types.js';
import { ApprovalMode } from '../policy/types.js';

vi.mock('../config/config.js');
vi.mock('../core/baseLlmClient.js');
vi.mock('./strategies/defaultStrategy.js');
vi.mock('./strategies/compositeStrategy.js');
vi.mock('./strategies/fallbackStrategy.js');
vi.mock('./strategies/overrideStrategy.js');
vi.mock('./strategies/classifierStrategy.js');
vi.mock('../telemetry/loggers.js');
vi.mock('../telemetry/types.js');

describe('ModelRouterService', () => {
  let service: ModelRouterService;
  let mockConfig: Config;
  let mockBaseLlmClient: BaseLlmClient;
  let mockContext: RoutingContext;
  let mockCompositeStrategy: CompositeStrategy;

  beforeEach(() => {
    vi.clearAllMocks();

    mockConfig = new Config({} as never);
    mockBaseLlmClient = {} as BaseLlmClient;
    vi.spyOn(mockConfig, 'getBaseLlmClient').mockReturnValue(mockBaseLlmClient);

    mockCompositeStrategy = new CompositeStrategy(
      [
        new FallbackStrategy(),
        new OverrideStrategy(),
        new ClassifierStrategy(),
        new DefaultStrategy(),
      ],
      'agent-router',
    );
    vi.mocked(CompositeStrategy).mockImplementation(
      () => mockCompositeStrategy,
    );

    service = new ModelRouterService(mockConfig);

    mockContext = {
      history: [],
      request: [{ text: 'test prompt' }],
      signal: new AbortController().signal,
    };
  });

  it('should initialize with a CompositeStrategy', () => {
    expect(CompositeStrategy).toHaveBeenCalled();
    expect(service['strategy']).toBeInstanceOf(CompositeStrategy);
  });

  it('should initialize the CompositeStrategy with the correct child strategies in order', () => {
    // This test relies on the mock implementation detail of the constructor
    const compositeStrategyArgs = vi.mocked(CompositeStrategy).mock.calls[0];
    const childStrategies = compositeStrategyArgs[0];

    expect(childStrategies.length).toBe(4);
    expect(childStrategies[0]).toBeInstanceOf(FallbackStrategy);
    expect(childStrategies[1]).toBeInstanceOf(OverrideStrategy);
    expect(childStrategies[2]).toBeInstanceOf(ClassifierStrategy);
    expect(childStrategies[3]).toBeInstanceOf(DefaultStrategy);
    expect(compositeStrategyArgs[1]).toBe('agent-router');
  });

  describe('route()', () => {
    const strategyDecision: RoutingDecision = {
      model: 'strategy-chosen-model',
      metadata: {
        source: 'test-router/fallback',
        latencyMs: 10,
        reasoning: 'Strategy reasoning',
        requiresVerification: false,
      },
    };

    it('should delegate routing to the composite strategy', async () => {
      const strategySpy = vi
        .spyOn(mockCompositeStrategy, 'route')
        .mockResolvedValue(strategyDecision);

      const decision = await service.route(mockContext);

      expect(strategySpy).toHaveBeenCalledWith(
        mockContext,
        mockConfig,
        mockBaseLlmClient,
      );
      expect(decision).toEqual(strategyDecision);
    });

    it('should log a telemetry event on a successful decision', async () => {
      vi.spyOn(mockCompositeStrategy, 'route').mockResolvedValue(
        strategyDecision,
      );

      await service.route(mockContext);

      expect(ModelRoutingEvent).toHaveBeenCalledWith(
        'strategy-chosen-model',
        'test-router/fallback',
        10,
        'Strategy reasoning',
        false,
        undefined,
        false,
        undefined,
      );
      expect(logModelRouting).toHaveBeenCalledWith(
        mockConfig,
        expect.any(ModelRoutingEvent),
      );
    });

    it('should log a telemetry event and re-throw on a failed decision', async () => {
      const testError = new Error('Strategy failed');
      vi.spyOn(mockCompositeStrategy, 'route').mockRejectedValue(testError);
      vi.spyOn(mockConfig, 'getModel').mockReturnValue('default-model');

      await expect(service.route(mockContext)).rejects.toThrow(testError);

      expect(ModelRoutingEvent).toHaveBeenCalledWith(
        'default-model',
        'router-exception',
        expect.any(Number),
        'An exception occurred during routing.',
        false,
        undefined,
        true,
        'Strategy failed',
      );
      expect(logModelRouting).toHaveBeenCalledWith(
        mockConfig,
        expect.any(ModelRoutingEvent),
      );
    });

    it('should require verification when approval mode is AUTO_EDIT', async () => {
      vi.spyOn(mockConfig, 'getApprovalMode').mockReturnValue(
        ApprovalMode.AUTO_EDIT,
      );
      const decisionWithoutVerification: RoutingDecision = {
        model: 'auto-edit-model',
        metadata: {
          source: 'test-router/default',
          latencyMs: 15,
          reasoning: 'auto edit enforcement',
          requiresVerification: false,
        },
      };
      vi.spyOn(mockCompositeStrategy, 'route').mockResolvedValue(
        decisionWithoutVerification,
      );

      const decision = await service.route(mockContext);

      expect(decision.metadata.requiresVerification).toBe(true);
      expect(decision.metadata.verificationReason).toBe('policy_auto_edit');
      expect(ModelRoutingEvent).toHaveBeenCalledWith(
        'auto-edit-model',
        'test-router/default',
        15,
        'auto edit enforcement',
        true,
        'policy_auto_edit',
        false,
        undefined,
      );
    });

    it('should merge verification reasons with policy-enforced ones', async () => {
      vi.spyOn(mockConfig, 'getApprovalMode').mockReturnValue(
        ApprovalMode.AUTO_EDIT,
      );
      const decisionWithReason: RoutingDecision = {
        model: 'pro-model',
        metadata: {
          source: 'classifier',
          latencyMs: 20,
          reasoning: 'classifier flagged complex task',
          requiresVerification: true,
          verificationReason: 'classifier_complexity',
        },
      };
      vi.spyOn(mockCompositeStrategy, 'route').mockResolvedValue(
        decisionWithReason,
      );

      const decision = await service.route(mockContext);

      expect(decision.metadata.requiresVerification).toBe(true);
      expect(decision.metadata.verificationReason).toBe(
        'classifier_complexity;policy_auto_edit',
      );
      expect(ModelRoutingEvent).toHaveBeenCalledWith(
        'pro-model',
        'classifier',
        20,
        'classifier flagged complex task',
        true,
        'classifier_complexity;policy_auto_edit',
        false,
        undefined,
      );
    });
  });
});
