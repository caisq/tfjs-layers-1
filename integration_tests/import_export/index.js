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

async function runExportModelDemo(artifactsDir, modelName, config) {
  console.log(tfc);  // DEBUG
  console.log(tfc.io);  // DEBUG
  console.log(tfl);  // DEBUG

  async function saveModelToLocalStorage() {
    const model =
        tfl.sequential({
            layers: [tfl.layers.dense({units: 1, inputShape: [100]})]});
    console.log(model);  // DEBUG
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});  // DEBUG
    await model.fit(tfc.ones([1, 100]), tfc.ones([1, 1]));  // DEBUG
    model.getWeights()[0].print();  // DEBUG
    console.log('Calling model.save');  // DEBUG
    const saveResult = await model.save(tfc.io.browserLocalStorage('myModel'));
    console.log('Prediction from saved model:');  // DEBUG
    tfc.tidy(() => {
      model.predict(tfc.ones([1, 100])).print();  // DEBUG
      console.log('Done saved model');  // DEBUG
    });
    console.log('saveResult:', saveResult);  // DEBUG
  }
  const localStorageSaveButton =
    document.getElementById('save-to-local-storage');
  localStorageSaveButton.addEventListener('click', saveModelToLocalStorage);

  async function loadModelFromLocalStorage() {
    console.log('Loading model...');  // DEBUG
    const model =
        await tfl.loadModel(tfc.io.browserLocalStorage('myModel'));
    console.log('Loaded model:', model);  // DEBUG
    model.getWeights()[0].print();  // DEBUG
    console.log('Prediction from loaded model:');  // DEBUG
    tfc.tidy(() => {
      model.predict(tfc.ones([1, 100])).print();  // DEBUG
    });
  }
  const localStorageLoadButton =
    document.getElementById('load-from-local-storage');
  localStorageLoadButton.addEventListener('click', loadModelFromLocalStorage);

  const filePrefixInput = document.getElementById('download-file-prefix');
  async function saveModelToDownloads() {
    const model =
        tfl.sequential({
            layers: [tfl.layers.dense({units: 1, inputShape: [100]})]});
    console.log(model);  // DEBUG
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});  // DEBUG
    await model.fit(tfc.ones([1, 100]), tfc.ones([1, 1]));  // DEBUG
    model.getWeights()[0].print();  // DEBUG
    console.log('Calling model.save with triggerDownloads().');  // DEBUG
    console.log('Prediction from downloaded model:');  // DEBUG
    tfc.tidy(() => {
      model.predict(tfc.ones([1, 100])).print();  // DEBUG
    });
    const saveResult = await model.save(
        tfc.io.triggerDownloads(filePrefixInput.value));
    console.log('Prediction from saved model:');  // DEBUG
    console.log('saveResult:', saveResult);  // DEBUG
  }
  const downloadModelButton = document.getElementById('download-model');
  downloadModelButton.addEventListener('click', saveModelToDownloads);

  const uploadJSONInput = document.getElementById('upload-json');
  const uploadWeightsInput = document.getElementById('upload-weights');
  async function loadModelFromUserSelectedFiles() {
    if (uploadJSONInput.files.length !== 1) {
      throw new Error('Select exactly one model JSON file first.');
    }
    if (uploadWeightsInput.files.length !== 1) {
      throw new Error('Select exactly one binary weights file first.');
    }
    const model = await tfl.loadModel(
        tfc.io.files([uploadJSONInput.files[0], uploadWeightsInput.files[0]]));
    console.log('Loaded model from file:', model);  // DEBUG
    console.log(
        'Prediction from model loaded from user-selected files:');  // DEBUG
    tfc.tidy(() => {
      model.predict(tfc.ones([1, 100])).print();  // DEBUG
    });
  }
  const uploadModelButton = document.getElementById('upload-model');
  uploadModelButton.addEventListener('click', loadModelFromUserSelectedFiles);

  // const uploadJSON = document.getElementById('upload-json');
  // const uploadWeights = document.getElementById('upload-weights');
  // const uploadModelButton = document.getElementById('upload-model');
  // console.log('uploadModelButton: ', uploadModelButton);  // DEBUG
  // uploadModelButton.addEventListener('click', () => {
  //   console.log(uploadJSON.files);
  //   if (uploadJSON.files.length === 1) {
  //     const reader = new FileReader();  // TODO(cais): Use singleton?
  //     reader.onloadend = (event) => {
  //       console.log('model.json load end: ' + event.target.result);  // DEBUG
  //     };
  //     console.log(uploadJSON.files[0].constructor.name);  // DEBUG
  //     reader.readAsText(uploadJSON.files[0]);
  //   }
  //   console.log(uploadWeights.files);
  //   if (uploadWeights.files.length === 1) {
  //     const reader = new FileReader();  // TODO(cais): Use singleton?
  //     reader.onloadend = (event) => {
  //       const buffer = event.target.result;
  //       const array = new Float32Array(buffer);
  //       console.log('Weights load end: length: ' + array.length);  // DEBUG
  //       console.log('Weights load end: element 0: ' + array[0]);   // DEBUG
  //     };
  //     reader.readAsArrayBuffer(uploadWeights.files[0]);
  //   }
  // });
}

runExportModelDemo();
