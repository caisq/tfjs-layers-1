/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/* Original source: keras/callbacks.py */

import {add, div, keep, mul, nextFrame, Scalar, Tensor, tidy} from '@tensorflow/tfjs-core';

import {getScalar} from './backend/state';
import {Container} from './engine/container';
import {Logs, resolveScalarsInLogs, UnresolvedLogs} from './logs';
import * as generic_utils from './utils/generic_utils';
import {mean} from './utils/math_utils';


export type Params = {
  [key: string]: number|string|boolean|number[]|string[]|boolean[];
};

/**
 * Abstract base class used to build new callbacks.
 *
 * The `logs` dictionary that callback methods take as argument will contain
 * keys for quantities relevant to the current batch or epoch.
 *
 * Currently, the `.fit()` method of the `Sequential` model class
 * will include the following quantities in the `logs` that
 * it passes to its callbacks:
 *
 * onEpochEnd: Logs include `acc` and `loss`, and optionally include `valLoss`
 *   (if validation is enabled in `fit`), and `valAcc` (if validation and
 *   accuracy monitoring are enabled).
 * onBatchBegin: Logs include `size`, the number of samples in the current
 *   batch.
 * onBatchEnd: Logs include `loss`, and optionally `acc` (if accuracy monitoring
 *   is enabled).
 */
export abstract class BaseCallback {
  // TODO(michaelterry): This type is a best guess.
  validationData: Tensor|Tensor[] = null;
  /**
   * Training parameters (eg. verbosity, batch size, number of epochs...).
   */
  params: Params;

  setParams(params: Params): void {
    this.params = params;
  }

  async onEpochBegin(epoch: number, logs?: UnresolvedLogs) {}

  async onEpochEnd(epoch: number, logs?: UnresolvedLogs) {}

  async onBatchBegin(batch: number, logs?: UnresolvedLogs) {}

  async onBatchEnd(batch: number, logs?: UnresolvedLogs) {}

  async onTrainBegin(logs?: UnresolvedLogs) {}

  async onTrainEnd(logs?: UnresolvedLogs) {}

  // Model needs to call Callback.setModel(), but cannot actually depend on
  // Callback because that creates a cyclic dependency.  Providing this no-op
  // method on BaseCallback breaks the cycle: this way Model can depend on
  // BaseCallback but not on Callback.  The argument is typed as `Container`
  // (the superclass of Model) to avoid recapitulating the cycle.  Callback
  // overrides this method and enforces that the argument is really a Model.
  setModel(model: Container): void {
    // Do nothing. Use Callback instead of BaseCallback to track the model.
  }
}

/**
 * Container abstracting a list of callbacks.
 */
export class CallbackList {
  callbacks: BaseCallback[];
  queueLength: number;

  // TODO(cais): When the need arises, uncomment the following lines and
  // implement the queue for time values.
  // private deltaTBatch: number;
  // private deltaTsBatchBegin: Array<number>;
  // private deltaTsBatchEnd: Array<number>;

  /**
   * Constructor of CallbackList.
   * @param callbacks Array of `Callback` instances.
   * @param queueLength Queue length for keeping running statistics over
   *   callback execution time.
   */
  constructor(callbacks?: BaseCallback[], queueLength = 10) {
    // TODO(cais): Make use of queueLength when implementing the queue for time
    // values.
    if (callbacks == null) {
      callbacks = [];
    }
    this.callbacks = callbacks;
    this.queueLength = queueLength;
  }

  append(callback: BaseCallback): void {
    this.callbacks.push(callback);
  }

  setParams(params: Params): void {
    for (const callback of this.callbacks) {
      callback.setParams(params);
    }
  }

  setModel(model: Container): void {
    for (const callback of this.callbacks) {
      callback.setModel(model);
    }
  }

  /**
   * Called at the start of an epoch.
   * @param epoch Index of epoch.
   * @param logs Dictionary of logs.
   */
  async onEpochBegin(epoch: number, logs?: UnresolvedLogs) {
    if (logs == null) {
      logs = {};
    }
    for (const callback of this.callbacks) {
      await callback.onEpochBegin(epoch, logs);
    }
  }

  /**
   * Called at the end of an epoch.
   * @param epoch Index of epoch.
   * @param logs Dictionary of logs.
   */
  async onEpochEnd(epoch: number, logs?: UnresolvedLogs) {
    if (logs == null) {
      logs = {};
    }
    for (const callback of this.callbacks) {
      await callback.onEpochEnd(epoch, logs);
    }
  }

  /**
   * Called  right before processing a batch.
   * @param batch Index of batch within the current epoch.
   * @param logs Dictionary of logs.
   */
  async onBatchBegin(batch: number, logs?: UnresolvedLogs) {
    if (logs == null) {
      logs = {};
    }
    for (const callback of this.callbacks) {
      await callback.onBatchBegin(batch, logs);
    }
  }

  /**
   * Called at the end of a batch.
   * @param batch Index of batch within the current epoch.
   * @param logs Dictionary of logs.
   */
  async onBatchEnd(batch: number, logs?: UnresolvedLogs) {
    if (logs == null) {
      logs = {};
    }
    for (const callback of this.callbacks) {
      await callback.onBatchEnd(batch, logs);
    }
  }

  /**
   * Called at the beginning of training.
   * @param logs Dictionary of logs.
   */
  async onTrainBegin(logs?: UnresolvedLogs) {
    if (logs == null) {
      logs = {};
    }
    for (const callback of this.callbacks) {
      await callback.onTrainBegin(logs);
    }
  }

  /**
   * Called at the end of training.
   * @param logs Dictionary of logs.
   */
  async onTrainEnd(logs?: UnresolvedLogs) {
    if (logs == null) {
      logs = {};
    }
    for (const callback of this.callbacks) {
      await callback.onTrainEnd(logs);
    }
  }
}

/**
 * A class that manages when to yield the thread during model training.
 *
 *
 */
export class ModelTrainingYielder {
  readonly AUTO_YIELD_THRESHOLD_MILLIS = 16;
  readonly AUTO_YIELD_DECISION_BATCH_COUNT = 2;

  private batchCount: number;
  private lastYieldBatchCount: number;
  private batchStartMillis: number;
  private batchDurationsMillis: number[];
  private autoYieldEveryBatches: number;

  constructor() {
    this.batchCount = 0;
    this.batchDurationsMillis = [];
    this.autoYieldEveryBatches = null;
    this.batchStartMillis = Date.now();
  }

  /**
   * Find the first Scalar tensor in `logs` and await data() on it.
   */
  private async resolveOneTensorInLogs(logs: UnresolvedLogs) {
    for (const key in logs) {
      const value = logs[key];
      if (typeof value !== 'number') {
        await (value as Scalar).data();
        break;
      }
    }
  }

  /**
   * The action taken at the end of every batch.
   *
   * @param logs The logs from the batch.
   */
  async maybeYieldOnBatch(logs: UnresolvedLogs) {
    this.batchCount++;
    if (this.autoYieldEveryBatches == null) {
      await this.resolveOneTensorInLogs(logs);
      await nextFrame();
      const t = Date.now();
      console.log(
          `AutoYielder: onBatchEnd: batch #${this.batchCount} ` +
          `took ${t - this.batchStartMillis} ms.`);  // DEBUG
      if (this.batchCount > 1) {
        this.batchDurationsMillis.push(t - this.batchStartMillis);
        console.log(this.batchDurationsMillis);
        if (this.batchDurationsMillis.length >=
            this.AUTO_YIELD_DECISION_BATCH_COUNT) {
          this.autoYieldEveryBatches = Math.floor(
              this.AUTO_YIELD_THRESHOLD_MILLIS /
              mean(this.batchDurationsMillis));
          if (this.autoYieldEveryBatches < 1) {
            this.autoYieldEveryBatches = 1;
          }
          console.log('autoYieldEveryBatches:', this.autoYieldEveryBatches);
        }
      }
      this.batchStartMillis = t;
      this.lastYieldBatchCount = this.batchCount;
    } else {
      if (this.batchCount - this.lastYieldBatchCount >=
          this.autoYieldEveryBatches) {
        await this.resolveOneTensorInLogs(logs);
        await nextFrame();
        this.lastYieldBatchCount = this.batchCount;
      }
    }
  }
}

/**
 * Callback that accumulates epoch averages of metrics.
 *
 * This callback is automatically applied to every Model.
 */
export class BaseLogger extends BaseCallback {
  private seen: number;
  private totals: UnresolvedLogs;
  private autoYielder: ModelTrainingYielder;

  constructor() {
    super();
  }

  async onTrainBegin(logs?: UnresolvedLogs) {
    this.autoYielder = new ModelTrainingYielder();
  }

  async onEpochBegin(epoch: number) {
    this.seen = 0;
    this.totals = {};
  }

  async onBatchEnd(batch: number, logs?: UnresolvedLogs) {
    // TODO(cais): Use `if` statement for yieldEvery === 'auto';
    await this.autoYielder.maybeYieldOnBatch(logs);

    if (logs == null) {
      logs = {};
    }
    const batchSize = logs['size'] == null ? 0 : logs['size'] as number;
    this.seen += batchSize;
    for (const key in logs) {
      const value = logs[key];
      if (typeof value === 'number') {
        if (!this.totals.hasOwnProperty(key)) {
          this.totals[key] = 0;
        }
        this.totals[key] = this.totals[key] as number + value * batchSize;
      } else {
        let oldTotalsToDispose: Scalar;
        if (key in this.totals) {
          oldTotalsToDispose = this.totals[key] as Scalar;
        } else {
          this.totals[key] = getScalar(0);
        }

        this.totals[key] = tidy(
            () => add((this.totals[key] as Scalar),
                      mul(value, getScalar(batchSize))) as Scalar);
        if (oldTotalsToDispose != null) {
          oldTotalsToDispose.dispose();
        }
      }
    }
  }

  async onEpochEnd(epoch: number, logs?: UnresolvedLogs) {
    if (logs != null) {
      for (const key of this.params['metrics'] as string[]) {
        if (this.totals[key] == null) {
          continue;
        }
        if (typeof this.totals[key] === 'number') {
          logs[key] = this.totals[key] as number / this.seen;
        } else {
          tidy(() => {
            logs[key] = mul(div(getScalar(1), getScalar(this.seen)) as Scalar,
                            this.totals[key] as Scalar) as Scalar;
            (this.totals[key] as Tensor).dispose();
            keep(logs[key] as Scalar);
          });
        }
      }
    }
  }
}

/**
 * Callback that records events into a `History` object. This callback is
 * automatically applied to every TF.js Layers model. The `History` object gets
 * returned by the `fit` method of models.
 */
export class History extends BaseCallback {
  epoch: number[];
  history: {[key: string]: Array<number|Tensor>};

  async onTrainBegin(logs?: UnresolvedLogs) {
    this.epoch = [];
    this.history = {};
  }

  async onEpochEnd(epoch: number, logs?: UnresolvedLogs) {
    if (logs == null) {
      logs = {};
    }
    this.epoch.push(epoch);
    for (const key in logs) {
      if (this.history[key] == null) {
        this.history[key] = [];
      }
      this.history[key].push(logs[key]);
    }
  }

  /**
   * Await the values of all losses and metrics.
   */
  async syncData() {
    const promises: Array<Promise<Float32Array|Int32Array|Uint8Array>> = [];
    const keys: string[] = [];
    const indices: number[] = [];
    for (const key in this.history) {
      const valueArray = this.history[key];
      for (let i = 0; i < valueArray.length; ++i) {
        if (typeof valueArray[i] !== 'number') {
          const valueScalar = valueArray[i] as Tensor;
          promises.push(valueScalar.data());
          keys.push(key);
          indices.push(i);
        }
      }
    }
    const values = await Promise.all(promises);
    for (let n = 0; n < values.length; ++n) {
      const tensorToDispose = this.history[keys[n]][indices[n]] as Tensor;
      tensorToDispose.dispose();
      this.history[keys[n]][indices[n]] = values[n][0];
    }
  }
}

export interface CustomCallbackConfig {
  onTrainBegin?: (logs?: Logs) => Promise<void>;
  onTrainEnd?: (logs?: Logs) => Promise<void>;
  onEpochBegin?: (epoch: number, logs?: Logs) => Promise<void>;
  onEpochEnd?: (epoch: number, logs?: Logs) => Promise<void>;
  onBatchBegin?: (batch: number, logs?: Logs) => Promise<void>;
  onBatchEnd?: (batch: number, logs?: Logs) => Promise<void>;
}

/**
 * Custom callback for training.
 */
export class CustomCallback extends BaseCallback {
  protected readonly trainBegin: (logs?: Logs) => Promise<void>;
  protected readonly trainEnd: (logs?: Logs) => Promise<void>;
  protected readonly epochBegin: (epoch: number, logs?: Logs) => Promise<void>;
  protected readonly epochEnd: (epoch: number, logs?: Logs) => Promise<void>;
  protected readonly batchBegin: (batch: number, logs?: Logs) => Promise<void>;
  protected readonly batchEnd: (batch: number, logs?: Logs) => Promise<void>;

  constructor(config: CustomCallbackConfig) {
    super();
    this.trainBegin = config.onTrainBegin;
    this.trainEnd = config.onTrainEnd;
    this.epochBegin = config.onEpochBegin;
    this.epochEnd = config.onEpochEnd;
    this.batchBegin = config.onBatchBegin;
    this.batchEnd = config.onBatchEnd;
  }

  async onEpochBegin(epoch: number, logs?: UnresolvedLogs): Promise<void> {
    if (this.epochBegin != null) {
      await resolveScalarsInLogs(logs);
      await this.epochBegin(epoch, logs as Logs);
    }
  }

  async onEpochEnd(epoch: number, logs?: UnresolvedLogs): Promise<void> {
    if (this.epochEnd != null) {
      await resolveScalarsInLogs(logs);
      await this.epochEnd(epoch, logs as Logs);
    }
  }

  async onBatchBegin(batch: number, logs?: UnresolvedLogs): Promise<void> {
    if (this.batchBegin != null) {
      await resolveScalarsInLogs(logs);
      await this.batchBegin(batch, logs as Logs);
    }
  }

  async onBatchEnd(batch: number, logs?: UnresolvedLogs): Promise<void> {
    if (this.batchEnd != null) {
      await resolveScalarsInLogs(logs);
      await this.batchEnd(batch, logs as Logs);
    }
  }

  async onTrainBegin(logs?: UnresolvedLogs): Promise<void> {
    if (this.trainBegin != null) {
      await resolveScalarsInLogs(logs);
      await this.trainBegin(logs as Logs);
    }
  }

  async onTrainEnd(logs?: UnresolvedLogs): Promise<void> {
    if (this.trainEnd != null) {
      await resolveScalarsInLogs(logs);
      await this.trainEnd(logs as Logs);
    }
  }
}

/**
 * Standardize callbacks or configurations of them to an Array of callbacks.
 */
export function standardizeCallbacks(callbacks: BaseCallback|BaseCallback[]|
                                     CustomCallbackConfig|
                                     CustomCallbackConfig[]): BaseCallback[] {
  if (callbacks == null) {
    return null;
  }
  if (callbacks instanceof BaseCallback) {
    return [callbacks as BaseCallback];
  }
  if (Array.isArray(callbacks) && callbacks[0] instanceof BaseCallback) {
    return callbacks as BaseCallback[];
  }
  // Convert custom callback configs to custom callback objects.
  const callbackConfigs =
      generic_utils.toList(callbacks) as CustomCallbackConfig[];
  return callbackConfigs.map(
      callbackConfig => new CustomCallback(callbackConfig));
}
