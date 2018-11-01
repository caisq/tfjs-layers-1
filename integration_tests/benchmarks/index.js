/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import * as tfc from '@tensorflow/tfjs-core';
import * as tfl from '@tensorflow/tfjs-layers';
import * as ui from './ui';

async function runBenchmark(artifactsDir, modelName, config) {
  const modelPath = artifactsDir + modelName + '/';
  console.log('Loading model "' + modelName + '" and benchmark data...');
  const model = await tfl.loadModel(modelPath + 'model.json');
  console.log('Done loading model "' + modelName + '" and benchmark data.');

  const benchmarkData = await (await fetch(modelPath + 'data.json')).json();

  const lossMap = {
    mean_squared_error: 'meanSquaredError',
    categorical_crossentropy: 'categoricalCrossentropy',
  };
  // TODO(cais): Maybe TF.js Layers should tolerate these Python-style names
  // for losses.

  const batchSize = benchmarkData.batch_size;
  let xs;
  if (benchmarkData.input_shape == null) {
    xs = [];
    for (const input of model.inputs) {
      xs.push(tfc.randomUniform([batchSize].concat(input.shape.slice(1))));
    }
  } else {
    xs = tfc.randomUniform([batchSize].concat(benchmarkData.input_shape));
  }

  let ys;
  if (benchmarkData.target_shape == null) {
    ys = [];
    for (const output of model.outputs) {
      ys.push(tfc.randomUniform([batchSize].concat(output.shape.slice(1))));
    }
  } else {
    ys = tfc.randomUniform([batchSize].concat(benchmarkData.target_shape));
  }

  if (benchmarkData.train_epochs > 0) {
    model.compile({
      optimizer: benchmarkData.optimizer,
      loss: lossMap[benchmarkData.loss],
    });
  }

  const FIT_BURNIN_EPOCHS = config.FIT_BURNIN_EPOCHS;
  const PREDICT_BURNINS = config.PREDICT_BURNINS;
  const PREDICT_RUNS = config.PREDICT_RUNS;

  // Perform fit() burn-in.
  if (benchmarkData.train_epochs > 0) {
    await model.fit(xs, ys, {
      batchSize: benchmarkData.batch_size,
      epochs: FIT_BURNIN_EPOCHS,
      yieldEvery: 'never'
    });
    model.trainableWeights[0].read().dataSync();
  }

  let trainTimeMs;
  if (benchmarkData.train_epochs > 0) {
    const trainBeginMs = performance.now();
    await model.fit(xs, ys, {
      batchSize: benchmarkData.batch_size,
      epochs: benchmarkData.train_epochs,
      yieldEvery: 'never'
    });
    // After the fit() call, call dataSync() to let the scheduled GPU
    // operations to complete before proceeding.
    model.trainableWeights[0].read().dataSync();
    const trainEndMs = performance.now();
    trainTimeMs = (trainEndMs - trainBeginMs) / benchmarkData.train_epochs;
  }

  // Perform predict() burn-in.
  return tfc.tidy(() => {
    let output;
    for (let i = 0; i < PREDICT_BURNINS; ++i) {
      output = tfc.tidy(() => {
        return model.predict(xs);
      });
    }
    // Time predict() a number of times and take the average.
    const predictBeginMs = performance.now();
    for (let i = 0; i < PREDICT_RUNS; ++i) {
      output = tfc.tidy(() => {
        return model.predict(xs);
      });
    }
    // After all the model.predict() calls, invoke dataSync() once to let the
    // scheduled GPU operations complete before proceeding.
    if (Array.isArray(output)) {
      output.map(out => out.dataSync());
    } else {
      output.dataSync();
    }
    const predictEndMs = performance.now();
    const predictTimeMs = (predictEndMs - predictBeginMs) / PREDICT_RUNS;

    tfc.dispose(xs);
    tfc.dispose(ys);

    return {
      originalData: benchmarkData,
      predictTimeMs: predictTimeMs,
      trainTimeMs: trainTimeMs,
    };
  });
}

function getRunAllBenchmarks(artifactsDir, benchmarks) {
  const runAllBenchmarks = async () => {
    ui.status('Running benchmarks...');
    for (let i = 0; i < benchmarks.models.length; ++i) {
      const modelName = benchmarks.models[i];
      ui.status(
          'Running model (' + (i + 1) + ' of ' + benchmarks.models.length +
          '): "' + modelName +
          '" ... (Please wait patiently. Do NOT click anything.)');
      await tfc.nextFrame();
      console.log('Benchmarking model: ' + modelName);
      const result =
          await runBenchmark(artifactsDir, modelName, benchmarks.config);
      ui.addResult(modelName, result);
    }
    ui.status('Standing by.');
  };
  return runAllBenchmarks;
}

async function setupBenchmarks() {
  const artifactsDir = './dist/data/';

  console.log('Loading benchmarks...');
  const url = 'http:' + artifactsDir + 'benchmarks.json';
  console.log(url);
  const x = await fetch(url);
  const benchmarks = await x.json();
  console.log('Done loading benchmarks:', benchmarks);

  ui.setMetadata(benchmarks.metadata);
  ui.setRunBenchmarksFunction(getRunAllBenchmarks(artifactsDir, benchmarks));
}

setupBenchmarks();
