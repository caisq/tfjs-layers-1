/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Unit tests for Eager Models.
 */

// tslint:disable:max-line-length
import {ones, scalar, Tensor, zeros} from '@tensorflow/tfjs-core';

import {Dense} from '../layers/core';
import {DType, Shape, SymbolicTensor} from '../types';
import {getExactlyOneTensor} from '../utils/generic_utils';
import {describeMathGPU, expectTensorsClose} from '../utils/test_utils';

import {Layer} from './topology';
import {Model} from './training';

// tslint:enable:max-line-length

describeMathGPU('Eager Model: One input', () => {
  class EagerModelForTest extends Model {
    dense1: Layer;
    dense2: Layer;

    constructor() {
      super({});
      this.dense1 = new Dense({
        units: 5,
        name: 'dense1',
        kernelInitializer: 'ones',
        biasInitializer: 'ones'
      });
      this.dense2 = new Dense({
        units: 2,
        name: 'dense2',
        kernelInitializer: 'ones',
        biasInitializer: 'ones'
      });
    }

    // tslint:disable-next-line:no-any
    call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
      inputs = getExactlyOneTensor(inputs);
      return this.dense2.apply(this.dense1.apply(inputs)) as Tensor;
    }

    computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
      return this.dense2.computeOutputShape(
          this.dense1.computeOutputShape(inputShape));
    }
  }

  it('getLayer by name gives correct results', () => {
    const eagerModel = new EagerModelForTest();
    expect(eagerModel.getLayer(eagerModel.dense1.name)).toBe(eagerModel.dense1);
    expect(eagerModel.getLayer(eagerModel.dense2.name)).toBe(eagerModel.dense2);
  });

  it('getLayer with index throws error', () => {
    const eagerModel = new EagerModelForTest();
    expect(() => eagerModel.getLayer(null, 0))
        .toThrowError(/should not be called with an index number.*eager/);
  });

  it('apply() call with concrete Tensor input gives correct output', () => {
    const eagerModel = new EagerModelForTest();
    const x = ones([3, 5]);
    const y = eagerModel.apply(x) as Tensor;
    expectTensorsClose(y, ones([3, 2]).mul(scalar(31)));
  });

  it('apply() call with SymbolicTensor input gives correct output', () => {
    const eagerModel = new EagerModelForTest();
    let x = new SymbolicTensor(DType.float32, [8, 4], null, null, null);
    let y = eagerModel.apply(x) as SymbolicTensor;
    expect(y.shape).toEqual([8, 2]);
    x = new SymbolicTensor(DType.float32, [7, 4], null, null, null);
    y = eagerModel.apply(x) as SymbolicTensor;
    expect(y.shape).toEqual([7, 2]);
  });

  it('predict() gives correct output', () => {
    const eagerModel = new EagerModelForTest();
    let x = ones([3, 5]);
    let y = eagerModel.predict(x) as Tensor;
    expectTensorsClose(y, ones([3, 2]).mul(scalar(31)));
    x = zeros([4, 5]);
    y = eagerModel.predict(x) as Tensor;
    expectTensorsClose(y, ones([4, 2]).mul(scalar(6)));
  });

  it('fit() gives correct output', async done => {
    // Use the following code to generate golden values:
    // ```python
    // import keras
    // import numpy as np
    // import tensorflow as tf
    //
    // class EagerModel(tf.keras.Model):
    //
    //   def __init__(self):
    //     super(EagerModel, self).__init__()
    //     self.dense1 = tf.keras.layers.Dense(
    //         5, kernel_initializer='ones', bias_initializer='ones')
    //     self.dense2 = tf.keras.layers.Dense(
    //         2, kernel_initializer='ones', bias_initializer='ones')
    //
    //   def call(self, inputs):
    //     return self.dense2(self.dense1(inputs))
    //
    // tf.enable_eager_execution()
    // model = EagerModel()
    // model.compile(optimizer=tf.train.GradientDescentOptimizer(0.01),
    // loss='mean_squared_error')
    //
    // x = np.ones([3, 5])
    // y = np.ones([3, 2])
    // history = model.fit(x, y, epochs=5)
    // print(history.history)
    // ```
    const eagerModel = new EagerModelForTest();
    eagerModel.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    const x = ones([3, 5]);
    const y = ones([3, 2]);
    eagerModel.fit(x, y, {epochs: 5})
        .then(history => {
          console.log(history.history);
          expectTensorsClose(history.history.loss as number[], [
            900.0, 98.0100326538086, 18.295082092285156, 9.381156921386719,
            5.752208709716797
          ]);
          done();
        })
        .catch(err => {
          console.error(err.stack);
        });
  });

  // TODO(cais): Test model.fit() with metrics.

  it('evaluate() gives correct output', () => {
    console.log('====== BEGIN ======');  // DEBUG
    const eagerModel = new EagerModelForTest();
    eagerModel.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    const x = ones([3, 5]);
    const y = ones([3, 2]);
    const res = eagerModel.evaluate(x, y) as Tensor[];
    expectTensorsClose(res[0], scalar(900));
    // TODO(cais): This should be a singleton Scalar, not an Array.
    console.log(res);                  // DEBUG
    console.log('====== END ======');  // DEBUG
  });

  it('trainableWeights gives correct results', () => {
    const eagerModel = new EagerModelForTest();
    // Invokke the model once to let it build first.
    eagerModel.apply(zeros([3, 5]));
    expect(eagerModel.trainableWeights.length).toEqual(4);
  });

  it('nonTrainableWeights gives correct results', () => {
    const eagerModel = new EagerModelForTest();
    // Invokke the model once to let it build first.
    eagerModel.apply(zeros([3, 5]));
    expect(eagerModel.nonTrainableWeights).toEqual([]);
  });
});

describeMathGPU('Eager Model: Missing overrides', () => {
  class BadEagerModelForTest extends Model {
    dense1: Layer;
    dense2: Layer;

    constructor() {
      super({});
      this.dense1 = new Dense({
        units: 5,
        name: 'dense1',
        kernelInitializer: 'ones',
        biasInitializer: 'ones'
      });
      this.dense2 = new Dense({
        units: 2,
        name: 'dense2',
        kernelInitializer: 'ones',
        biasInitializer: 'ones'
      });
    }
  }

  it('apply() with Tensor throws error', () => {
    const model = new BadEagerModelForTest();
    const x = ones([3, 5]);
    expect(() => model.apply(x))
        .toThrowError(
            /call method of an eager Container needs to be specified/);
  });

  it('apply() with SymbolicTensor throws error', () => {
    const model = new BadEagerModelForTest();
    let x = new SymbolicTensor(DType.float32, [8, 4], null, null, null);
    expect(() => model.apply(x))
        .toThrowError(
            /computeOutputShape .* of an eager Container needs to be specified/);
  });
});
