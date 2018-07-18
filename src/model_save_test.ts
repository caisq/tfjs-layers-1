/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {io, randomNormal, Tensor, zeros} from '@tensorflow/tfjs-core';

import * as tfl from './index';
import {modelFromJSON} from './models';
// import {modelFromJSON} from './models';
// tslint:disable-next-line:max-line-length
import {describeMathCPUAndGPU, describeMathGPU, expectTensorsClose} from './utils/test_utils';

describeMathCPUAndGPU('Model.save', () => {
  class IOHandlerForTest implements io.IOHandler {
    savedArtifacts: io.ModelArtifacts;

    async save(modelArtifacts: io.ModelArtifacts): Promise<io.SaveResult> {
      this.savedArtifacts = modelArtifacts;
      return {modelArtifactsInfo: null};
    }
  }

  class EmptyIOHandler implements io.IOHandler {}

  it('Saving all weights succeeds', async done => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({units: 3, inputShape: [5]}));
    const handler = new IOHandlerForTest();

    model.save(handler)
        .then(saveResult => {
          expect(handler.savedArtifacts.modelTopology)
              .toEqual(model.toJSON(null, false));
          expect(handler.savedArtifacts.weightSpecs.length).toEqual(2);
          expect(handler.savedArtifacts.weightSpecs[0].name.indexOf('/kernel'))
              .toBeGreaterThan(0);
          expect(handler.savedArtifacts.weightSpecs[0].shape).toEqual([5, 3]);
          expect(handler.savedArtifacts.weightSpecs[0].dtype)
              .toEqual('float32');
          expect(handler.savedArtifacts.weightSpecs[1].name.indexOf('/bias'))
              .toBeGreaterThan(0);
          expect(handler.savedArtifacts.weightSpecs[1].shape).toEqual([3]);
          expect(handler.savedArtifacts.weightSpecs[1].dtype)
              .toEqual('float32');
          done();
        })
        .catch(err => {
          console.error(err.stack);
        });
  });

  it('Saving only trainable weights succeeds', async done => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({units: 3, inputShape: [5], trainable: false}));
    model.add(tfl.layers.dense({units: 2}));
    const handler = new IOHandlerForTest();

    model.save(handler, {trainableOnly: true})
        .then(saveResult => {
          expect(handler.savedArtifacts.modelTopology)
              .toEqual(model.toJSON(null, false));
          // Verify that only the trainable weights (i.e., weights from the
          // 2nd, trainable Dense layer) are saved.
          expect(handler.savedArtifacts.weightSpecs.length).toEqual(2);
          expect(handler.savedArtifacts.weightSpecs[0].name.indexOf('/kernel'))
              .toBeGreaterThan(0);
          expect(handler.savedArtifacts.weightSpecs[0].shape).toEqual([3, 2]);
          expect(handler.savedArtifacts.weightSpecs[0].dtype)
              .toEqual('float32');
          expect(handler.savedArtifacts.weightSpecs[1].name.indexOf('/bias'))
              .toBeGreaterThan(0);
          expect(handler.savedArtifacts.weightSpecs[1].shape).toEqual([2]);
          expect(handler.savedArtifacts.weightSpecs[1].dtype)
              .toEqual('float32');
          done();
        })
        .catch(err => {
          console.error(err.stack);
        });
  });

  it('Saving to a handler without save method fails', async done => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({units: 3, inputShape: [5]}));
    const handler = new EmptyIOHandler();
    model.save(handler)
        .then(saveResult => {
          fail(
              'Saving with an IOHandler without `save` succeeded ' +
              'unexpectedly.');
        })
        .catch(err => {
          expect(err.message)
              .toEqual(
                  'Model.save() cannot proceed because the IOHandler ' +
                  'provided does not have the `save` attribute defined.');
          done();
        });
  });
});

describeMathGPU('Save-load round trips', () => {
  it('Sequential model, Local storage', done => {
    const model1 = tfl.sequential();
    model1.add(
        tfl.layers.dense({units: 2, inputShape: [2], activation: 'relu'}));
    model1.add(tfl.layers.dense({units: 1, useBias: false}));

    // Use a randomly generated model path to prevent collision.
    const path = `testModel${new Date().getTime()}_${Math.random()}`;

    // First save the model to local storage.
    const modelURL = `localstorage://${path}`;
    model1.save(modelURL)
        .then(saveResult => {
          // Once the saving succeeds, load the model back.
          tfl.loadModel(modelURL)
              .then(model2 => {
                // Verify that the topology of the model is correct.
                expect(model2.toJSON(null, false))
                    .toEqual(model1.toJSON(null, false));

                // Check the equality of the two models' weights.
                const weights1 = model1.getWeights();
                const weights2 = model2.getWeights();
                expect(weights2.length).toEqual(weights1.length);
                for (let i = 0; i < weights1.length; ++i) {
                  expectTensorsClose(weights1[i], weights2[i]);
                }

                done();
              })
              .catch(err => {
                done.fail(err.stack);
              });
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('Functional model, IndexedDB', done => {
    const input = tfl.input({shape: [2, 2]});
    const layer1 = tfl.layers.flatten().apply(input);
    const layer2 =
        tfl.layers.dense({units: 2}).apply(layer1) as tfl.SymbolicTensor;
    const model1 = tfl.model({inputs: input, outputs: layer2});
    // Use a randomly generated model path to prevent collision.
    const path = `testModel${new Date().getTime()}_${Math.random()}`;

    // First save the model to local storage.
    const modelURL = `indexeddb://${path}`;
    model1.save(modelURL)
        .then(saveResult => {
          // Once the saving succeeds, load the model back.
          tfl.loadModel(modelURL)
              .then(model2 => {
                // Verify that the topology of the model is correct.
                expect(model2.toJSON(null, false))
                    .toEqual(model1.toJSON(null, false));

                // Check the equality of the two models' weights.
                const weights1 = model1.getWeights();
                const weights2 = model2.getWeights();
                expect(weights2.length).toEqual(weights1.length);
                for (let i = 0; i < weights1.length; ++i) {
                  expectTensorsClose(weights1[i], weights2[i]);
                }

                done();
              })
              .catch(err => {
                done.fail(err.stack);
              });
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('Call predict() and fit() after load: conv2d model', done => {
    const model = tfl.sequential();
    model.add(tfl.layers.conv2d({
      filters: 8,
      kernelSize: 4,
      inputShape: [28, 28, 1],
      padding: 'same',
      activation: 'relu'
    }));
    model.add(tfl.layers.maxPooling2d({
      poolSize: 2,
      padding: 'same',
    }));
    model.add(tfl.layers.flatten());
    model.add(tfl.layers.dense({units: 1}));

    const x = randomNormal([1, 28, 28, 1]);
    const y = model.predict(x) as Tensor;

    const path = `testModel${new Date().getTime()}_${Math.random()}`;
    const url = `indexeddb://${path}`;
    model.save(url)
        .then(saveResult => {
          // Load the model back.
          tfl.loadModel(url)
              .then(modelPrime => {
                // Call predict() on the loaded model and assert the result
                // equals the original predict() result.
                const yPrime = modelPrime.predict(x) as Tensor;
                expectTensorsClose(y, yPrime);

                // Call compile and fit() on the loaded model.
                modelPrime.compile(
                    {optimizer: 'sgd', loss: 'meanSquaredError'});
                const trainExamples = 10;
                modelPrime
                    .fit(
                        randomNormal([trainExamples, 28, 28, 1]),
                        randomNormal([trainExamples]), {epochs: 4})
                    .then(history => {
                      done();
                    })
                    .catch(err => done.fail(err.stack));
              })
              .catch(err => done.fail(err.stack));
        })
        .catch(err => done.fail(err.stack));
  });

  it('Call predict() and fit() after load: conv1d model', done => {
    const model = tfl.sequential();
    model.add(tfl.layers.conv1d({
      filters: 8,
      kernelSize: 4,
      inputShape: [100, 1],
      padding: 'same',
      activation: 'relu'
    }));
    model.add(tfl.layers.maxPooling1d({
      poolSize: 2,
      padding: 'same',
    }));
    model.add(tfl.layers.flatten());
    model.add(tfl.layers.dense({units: 1}));

    const x = randomNormal([1, 100, 1]);
    const y = model.predict(x) as Tensor;

    const path = `testModel${new Date().getTime()}_${Math.random()}`;
    const url = `indexeddb://${path}`;
    model.save(url)
        .then(saveResult => {
          // Load the model back.
          tfl.loadModel(url)
              .then(modelPrime => {
                // Call predict() on the loaded model and assert the
                // result equals the original predict() result.
                const yPrime = modelPrime.predict(x) as Tensor;
                expectTensorsClose(y, yPrime);

                // Call compile and fit() on the loaded model.
                modelPrime.compile(
                    {optimizer: 'sgd', loss: 'meanSquaredError'});
                const trainExamples = 10;
                modelPrime
                    .fit(
                        randomNormal([trainExamples, 100, 1]),
                        randomNormal([trainExamples]), {epochs: 4})
                    .then(history => {
                      done();
                    })
                    .catch(err => done.fail(err.stack));
              })
              .catch(err => done.fail(err.stack));
        })
        .catch(err => done.fail(err.stack));
  });

  it('Call predict() and fit() after load: Bidirectional LSTM', done => {
    const model = tfl.sequential();
    const lstmUnits = 3;
    const sequenceLength = 4;
    const inputDims = 5;
    model.add(tfl.layers.bidirectional({
      layer: tfl.layers.lstm({units: lstmUnits}) as tfl.RNN,
      mergeMode: 'concat',
      inputShape: [sequenceLength, inputDims]
    }));

    const x = randomNormal([2, 4, 5]);
    const y = model.predict(x) as Tensor;

    const path = `testModel${new Date().getTime()}_${Math.random()}`;
    const url = `indexeddb://${path}`;
    model.save(url)
        .then(saveResult => {
          tfl.loadModel(url)
              .then(modelPrime => {
                const yPrime = modelPrime.predict(x) as Tensor;
                expectTensorsClose(y, yPrime);

                // Call compile and fit() on the loaded model.
                modelPrime.compile(
                    {optimizer: 'sgd', loss: 'meanSquaredError'});
                const trainExamples = 2;
                modelPrime
                    .fit(
                        randomNormal(
                            [trainExamples, sequenceLength, inputDims]),
                        randomNormal([trainExamples, lstmUnits * 2]),
                        {epochs: 2})
                    .then(history => {
                      done();
                    })
                    .catch(err => done.fail(err.stack));
              })
              .catch(err => done.fail(err.stack));
        })
        .catch(err => done.fail(err.stack));
  });

  it('Bug model test', async done => {
    console.log('====== BEGIN ======');  // DEBUG
    const jsonString =
        `{"modelTopology": {"model_config": {"class_name": "Model", "config": {"input_layers": [["input_1", 0, 0]], "layers": [{"class_name": "InputLayer", "config": {"sparse": false, "dtype": "float32", "name": "input_1", "batch_input_shape": [null, null, null, 3]}, "inbound_nodes": [], "name": "input_1"}, {"class_name": "Conv2D", "config": {"use_bias": false, "activation": "linear", "bias_constraint": null, "dilation_rate": [1, 1], "bias_regularizer": null, "strides": [1, 1], "padding": "same", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0005000000237487257}}, "name": "conv2d_1", "data_format": "channels_last", "activity_regularizer": null, "kernel_size": [3, 3], "filters": 16, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "seed": null, "distribution": "uniform", "mode": "fan_avg"}}, "trainable": true}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "name": "conv2d_1"}, {"class_name": "BatchNormalization", "config": {"scale": true, "gamma_regularizer": null, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "center": true, "gamma_initializer": {"class_name": "Ones", "config": {}}, "gamma_constraint": null, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "epsilon": 0.001, "axis": -1, "name": "batch_normalization_1", "beta_regularizer": null, "trainable": true, "momentum": 0.99, "beta_initializer": {"class_name": "Zeros", "config": {}}, "beta_constraint": null}, "inbound_nodes": [[["conv2d_1", 0, 0, {}]]], "name": "batch_normalization_1"}, {"class_name": "LeakyReLU", "config": {"alpha": 0.10000000149011612, "name": "leaky_re_lu_1", "trainable": true}, "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]], "name": "leaky_re_lu_1"}, {"class_name": "MaxPooling2D", "config": {"pool_size": [2, 2], "strides": [2, 2], "trainable": true, "padding": "same", "data_format": "channels_last", "name": "max_pooling2d_1"}, "inbound_nodes": [[["leaky_re_lu_1", 0, 0, {}]]], "name": "max_pooling2d_1"}, {"class_name": "Conv2D", "config": {"use_bias": false, "activation": "linear", "bias_constraint": null, "dilation_rate": [1, 1], "bias_regularizer": null, "strides": [1, 1], "padding": "same", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0005000000237487257}}, "name": "conv2d_2", "data_format": "channels_last", "activity_regularizer": null, "kernel_size": [3, 3], "filters": 32, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "seed": null, "distribution": "uniform", "mode": "fan_avg"}}, "trainable": true}, "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]], "name": "conv2d_2"}, {"class_name": "BatchNormalization", "config": {"scale": true, "gamma_regularizer": null, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "center": true, "gamma_initializer": {"class_name": "Ones", "config": {}}, "gamma_constraint": null, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "epsilon": 0.001, "axis": -1, "name": "batch_normalization_2", "beta_regularizer": null, "trainable": true, "momentum": 0.99, "beta_initializer": {"class_name": "Zeros", "config": {}}, "beta_constraint": null}, "inbound_nodes": [[["conv2d_2", 0, 0, {}]]], "name": "batch_normalization_2"}, {"class_name": "LeakyReLU", "config": {"alpha": 0.10000000149011612, "name": "leaky_re_lu_2", "trainable": true}, "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]], "name": "leaky_re_lu_2"}, {"class_name": "MaxPooling2D", "config": {"pool_size": [2, 2], "strides": [2, 2], "trainable": true, "padding": "same", "data_format": "channels_last", "name": "max_pooling2d_2"}, "inbound_nodes": [[["leaky_re_lu_2", 0, 0, {}]]], "name": "max_pooling2d_2"}, {"class_name": "Conv2D", "config": {"use_bias": false, "activation": "linear", "bias_constraint": null, "dilation_rate": [1, 1], "bias_regularizer": null, "strides": [1, 1], "padding": "same", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0005000000237487257}}, "name": "conv2d_3", "data_format": "channels_last", "activity_regularizer": null, "kernel_size": [3, 3], "filters": 64, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "seed": null, "distribution": "uniform", "mode": "fan_avg"}}, "trainable": true}, "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]], "name": "conv2d_3"}, {"class_name": "BatchNormalization", "config": {"scale": true, "gamma_regularizer": null, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "center": true, "gamma_initializer": {"class_name": "Ones", "config": {}}, "gamma_constraint": null, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "epsilon": 0.001, "axis": -1, "name": "batch_normalization_3", "beta_regularizer": null, "trainable": true, "momentum": 0.99, "beta_initializer": {"class_name": "Zeros", "config": {}}, "beta_constraint": null}, "inbound_nodes": [[["conv2d_3", 0, 0, {}]]], "name": "batch_normalization_3"}, {"class_name": "LeakyReLU", "config": {"alpha": 0.10000000149011612, "name": "leaky_re_lu_3", "trainable": true}, "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]], "name": "leaky_re_lu_3"}, {"class_name": "MaxPooling2D", "config": {"pool_size": [2, 2], "strides": [2, 2], "trainable": true, "padding": "same", "data_format": "channels_last", "name": "max_pooling2d_3"}, "inbound_nodes": [[["leaky_re_lu_3", 0, 0, {}]]], "name": "max_pooling2d_3"}, {"class_name": "Conv2D", "config": {"use_bias": false, "activation": "linear", "bias_constraint": null, "dilation_rate": [1, 1], "bias_regularizer": null, "strides": [1, 1], "padding": "same", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0005000000237487257}}, "name": "conv2d_4", "data_format": "channels_last", "activity_regularizer": null, "kernel_size": [3, 3], "filters": 128, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "seed": null, "distribution": "uniform", "mode": "fan_avg"}}, "trainable": true}, "inbound_nodes": [[["max_pooling2d_3", 0, 0, {}]]], "name": "conv2d_4"}, {"class_name": "BatchNormalization", "config": {"scale": true, "gamma_regularizer": null, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "center": true, "gamma_initializer": {"class_name": "Ones", "config": {}}, "gamma_constraint": null, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "epsilon": 0.001, "axis": -1, "name": "batch_normalization_4", "beta_regularizer": null, "trainable": true, "momentum": 0.99, "beta_initializer": {"class_name": "Zeros", "config": {}}, "beta_constraint": null}, "inbound_nodes": [[["conv2d_4", 0, 0, {}]]], "name": "batch_normalization_4"}, {"class_name": "LeakyReLU", "config": {"alpha": 0.10000000149011612, "name": "leaky_re_lu_4", "trainable": true}, "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]], "name": "leaky_re_lu_4"}, {"class_name": "MaxPooling2D", "config": {"pool_size": [2, 2], "strides": [2, 2], "trainable": true, "padding": "same", "data_format": "channels_last", "name": "max_pooling2d_4"}, "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]], "name": "max_pooling2d_4"}, {"class_name": "Conv2D", "config": {"use_bias": false, "activation": "linear", "bias_constraint": null, "dilation_rate": [1, 1], "bias_regularizer": null, "strides": [1, 1], "padding": "same", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0005000000237487257}}, "name": "conv2d_5", "data_format": "channels_last", "activity_regularizer": null, "kernel_size": [3, 3], "filters": 256, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "seed": null, "distribution": "uniform", "mode": "fan_avg"}}, "trainable": true}, "inbound_nodes": [[["max_pooling2d_4", 0, 0, {}]]], "name": "conv2d_5"}, {"class_name": "BatchNormalization", "config": {"scale": true, "gamma_regularizer": null, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "center": true, "gamma_initializer": {"class_name": "Ones", "config": {}}, "gamma_constraint": null, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "epsilon": 0.001, "axis": -1, "name": "batch_normalization_5", "beta_regularizer": null, "trainable": true, "momentum": 0.99, "beta_initializer": {"class_name": "Zeros", "config": {}}, "beta_constraint": null}, "inbound_nodes": [[["conv2d_5", 0, 0, {}]]], "name": "batch_normalization_5"}, {"class_name": "LeakyReLU", "config": {"alpha": 0.10000000149011612, "name": "leaky_re_lu_5", "trainable": true}, "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]], "name": "leaky_re_lu_5"}, {"class_name": "MaxPooling2D", "config": {"pool_size": [2, 2], "strides": [2, 2], "trainable": true, "padding": "same", "data_format": "channels_last", "name": "max_pooling2d_5"}, "inbound_nodes": [[["leaky_re_lu_5", 0, 0, {}]]], "name": "max_pooling2d_5"}, {"class_name": "Conv2D", "config": {"use_bias": false, "activation": "linear", "bias_constraint": null, "dilation_rate": [1, 1], "bias_regularizer": null, "strides": [1, 1], "padding": "same", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0005000000237487257}}, "name": "conv2d_6", "data_format": "channels_last", "activity_regularizer": null, "kernel_size": [3, 3], "filters": 512, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "seed": null, "distribution": "uniform", "mode": "fan_avg"}}, "trainable": true}, "inbound_nodes": [[["max_pooling2d_5", 0, 0, {}]]], "name": "conv2d_6"}, {"class_name": "BatchNormalization", "config": {"scale": true, "gamma_regularizer": null, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "center": true, "gamma_initializer": {"class_name": "Ones", "config": {}}, "gamma_constraint": null, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "epsilon": 0.001, "axis": -1, "name": "batch_normalization_6", "beta_regularizer": null, "trainable": true, "momentum": 0.99, "beta_initializer": {"class_name": "Zeros", "config": {}}, "beta_constraint": null}, "inbound_nodes": [[["conv2d_6", 0, 0, {}]]], "name": "batch_normalization_6"}, {"class_name": "LeakyReLU", "config": {"alpha": 0.10000000149011612, "name": "leaky_re_lu_6", "trainable": true}, "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]], "name": "leaky_re_lu_6"}, {"class_name": "MaxPooling2D", "config": {"pool_size": [2, 2], "strides": [1, 1], "trainable": true, "padding": "same", "data_format": "channels_last", "name": "max_pooling2d_6"}, "inbound_nodes": [[["leaky_re_lu_6", 0, 0, {}]]], "name": "max_pooling2d_6"}, {"class_name": "Conv2D", "config": {"use_bias": false, "activation": "linear", "bias_constraint": null, "dilation_rate": [1, 1], "bias_regularizer": null, "strides": [1, 1], "padding": "same", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0005000000237487257}}, "name": "conv2d_7", "data_format": "channels_last", "activity_regularizer": null, "kernel_size": [3, 3], "filters": 1024, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "seed": null, "distribution": "uniform", "mode": "fan_avg"}}, "trainable": true}, "inbound_nodes": [[["max_pooling2d_6", 0, 0, {}]]], "name": "conv2d_7"}, {"class_name": "BatchNormalization", "config": {"scale": true, "gamma_regularizer": null, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "center": true, "gamma_initializer": {"class_name": "Ones", "config": {}}, "gamma_constraint": null, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "epsilon": 0.001, "axis": -1, "name": "batch_normalization_7", "beta_regularizer": null, "trainable": true, "momentum": 0.99, "beta_initializer": {"class_name": "Zeros", "config": {}}, "beta_constraint": null}, "inbound_nodes": [[["conv2d_7", 0, 0, {}]]], "name": "batch_normalization_7"}, {"class_name": "LeakyReLU", "config": {"alpha": 0.10000000149011612, "name": "leaky_re_lu_7", "trainable": true}, "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]], "name": "leaky_re_lu_7"}, {"class_name": "Conv2D", "config": {"use_bias": false, "activation": "linear", "bias_constraint": null, "dilation_rate": [1, 1], "bias_regularizer": null, "strides": [1, 1], "padding": "same", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0005000000237487257}}, "name": "conv2d_8", "data_format": "channels_last", "activity_regularizer": null, "kernel_size": [1, 1], "filters": 256, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "seed": null, "distribution": "uniform", "mode": "fan_avg"}}, "trainable": true}, "inbound_nodes": [[["leaky_re_lu_7", 0, 0, {}]]], "name": "conv2d_8"}, {"class_name": "BatchNormalization", "config": {"scale": true, "gamma_regularizer": null, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "center": true, "gamma_initializer": {"class_name": "Ones", "config": {}}, "gamma_constraint": null, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "epsilon": 0.001, "axis": -1, "name": "batch_normalization_8", "beta_regularizer": null, "trainable": true, "momentum": 0.99, "beta_initializer": {"class_name": "Zeros", "config": {}}, "beta_constraint": null}, "inbound_nodes": [[["conv2d_8", 0, 0, {}]]], "name": "batch_normalization_8"}, {"class_name": "LeakyReLU", "config": {"alpha": 0.10000000149011612, "name": "leaky_re_lu_8", "trainable": true}, "inbound_nodes": [[["batch_normalization_8", 0, 0, {}]]], "name": "leaky_re_lu_8"}, {"class_name": "Conv2D", "config": {"use_bias": false, "activation": "linear", "bias_constraint": null, "dilation_rate": [1, 1], "bias_regularizer": null, "strides": [1, 1], "padding": "same", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0005000000237487257}}, "name": "conv2d_11", "data_format": "channels_last", "activity_regularizer": null, "kernel_size": [1, 1], "filters": 128, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "seed": null, "distribution": "uniform", "mode": "fan_avg"}}, "trainable": true}, "inbound_nodes": [[["leaky_re_lu_8", 0, 0, {}]]], "name": "conv2d_11"}, {"class_name": "BatchNormalization", "config": {"scale": true, "gamma_regularizer": null, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "center": true, "gamma_initializer": {"class_name": "Ones", "config": {}}, "gamma_constraint": null, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "epsilon": 0.001, "axis": -1, "name": "batch_normalization_10", "beta_regularizer": null, "trainable": true, "momentum": 0.99, "beta_initializer": {"class_name": "Zeros", "config": {}}, "beta_constraint": null}, "inbound_nodes": [[["conv2d_11", 0, 0, {}]]], "name": "batch_normalization_10"}, {"class_name": "LeakyReLU", "config": {"alpha": 0.10000000149011612, "name": "leaky_re_lu_10", "trainable": true}, "inbound_nodes": [[["batch_normalization_10", 0, 0, {}]]], "name": "leaky_re_lu_10"}, {"class_name": "UpSampling2D", "config": {"data_format": "channels_last", "name": "up_sampling2d_1", "size": [2, 2], "trainable": true}, "inbound_nodes": [[["leaky_re_lu_10", 0, 0, {}]]], "name": "up_sampling2d_1"}, {"class_name": "Concatenate", "config": {"axis": -1, "name": "concatenate_1", "trainable": true}, "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}], ["leaky_re_lu_5", 0, 0, {}]]], "name": "concatenate_1"}, {"class_name": "Conv2D", "config": {"use_bias": false, "activation": "linear", "bias_constraint": null, "dilation_rate": [1, 1], "bias_regularizer": null, "strides": [1, 1], "padding": "same", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0005000000237487257}}, "name": "conv2d_9", "data_format": "channels_last", "activity_regularizer": null, "kernel_size": [3, 3], "filters": 512, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "seed": null, "distribution": "uniform", "mode": "fan_avg"}}, "trainable": true}, "inbound_nodes": [[["leaky_re_lu_8", 0, 0, {}]]], "name": "conv2d_9"}, {"class_name": "Conv2D", "config": {"use_bias": false, "activation": "linear", "bias_constraint": null, "dilation_rate": [1, 1], "bias_regularizer": null, "strides": [1, 1], "padding": "same", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0005000000237487257}}, "name": "conv2d_12", "data_format": "channels_last", "activity_regularizer": null, "kernel_size": [3, 3], "filters": 256, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "seed": null, "distribution": "uniform", "mode": "fan_avg"}}, "trainable": true}, "inbound_nodes": [[["concatenate_1", 0, 0, {}]]], "name": "conv2d_12"}, {"class_name": "BatchNormalization", "config": {"scale": true, "gamma_regularizer": null, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "center": true, "gamma_initializer": {"class_name": "Ones", "config": {}}, "gamma_constraint": null, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "epsilon": 0.001, "axis": -1, "name": "batch_normalization_9", "beta_regularizer": null, "trainable": true, "momentum": 0.99, "beta_initializer": {"class_name": "Zeros", "config": {}}, "beta_constraint": null}, "inbound_nodes": [[["conv2d_9", 0, 0, {}]]], "name": "batch_normalization_9"}, {"class_name": "BatchNormalization", "config": {"scale": true, "gamma_regularizer": null, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "center": true, "gamma_initializer": {"class_name": "Ones", "config": {}}, "gamma_constraint": null, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "epsilon": 0.001, "axis": -1, "name": "batch_normalization_11", "beta_regularizer": null, "trainable": true, "momentum": 0.99, "beta_initializer": {"class_name": "Zeros", "config": {}}, "beta_constraint": null}, "inbound_nodes": [[["conv2d_12", 0, 0, {}]]], "name": "batch_normalization_11"}, {"class_name": "LeakyReLU", "config": {"alpha": 0.10000000149011612, "name": "leaky_re_lu_9", "trainable": true}, "inbound_nodes": [[["batch_normalization_9", 0, 0, {}]]], "name": "leaky_re_lu_9"}, {"class_name": "LeakyReLU", "config": {"alpha": 0.10000000149011612, "name": "leaky_re_lu_11", "trainable": true}, "inbound_nodes": [[["batch_normalization_11", 0, 0, {}]]], "name": "leaky_re_lu_11"}, {"class_name": "Conv2D", "config": {"use_bias": true, "activation": "linear", "bias_constraint": null, "dilation_rate": [1, 1], "bias_regularizer": null, "strides": [1, 1], "padding": "same", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0005000000237487257}}, "name": "conv2d_10", "data_format": "channels_last", "activity_regularizer": null, "kernel_size": [1, 1], "filters": 255, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "seed": null, "distribution": "uniform", "mode": "fan_avg"}}, "trainable": true}, "inbound_nodes": [[["leaky_re_lu_9", 0, 0, {}]]], "name": "conv2d_10"}, {"class_name": "Conv2D", "config": {"use_bias": true, "activation": "linear", "bias_constraint": null, "dilation_rate": [1, 1], "bias_regularizer": null, "strides": [1, 1], "padding": "same", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0005000000237487257}}, "name": "conv2d_13", "data_format": "channels_last", "activity_regularizer": null, "kernel_size": [1, 1], "filters": 255, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "seed": null, "distribution": "uniform", "mode": "fan_avg"}}, "trainable": true}, "inbound_nodes": [[["leaky_re_lu_11", 0, 0, {}]]], "name": "conv2d_13"}], "name": "model_1", "output_layers": [["conv2d_10", 0, 0], ["conv2d_13", 0, 0]]}}, "keras_version": "2.1.6", "backend": "tensorflow"}}`;
    try {
      // console.log(jsonString);  // DEBUG
      // console.log(JSON.parse(jsonString));
      const model = await modelFromJSON(JSON.parse(jsonString));
      model.summary();
      // (model.predict(zeros([1, 64, 64, 3])) as Tensor).print();
      console.log(model.predict(zeros([1, 64, 64, 3])));  // DEBUG
    } catch (err) {
      console.log(err.message);
      console.log(err.stack);
      done.fail();
    }
    done();
    console.log('====== END ======');  // DEBUG
  });
});
