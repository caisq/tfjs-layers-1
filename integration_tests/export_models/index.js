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
  const tf = tfl;
  const model = tf.sequential({
    layers: [tf.layers.dense({units: 1, inputShape: [100]})]
  });
  const data = await model.save();

  const downloadJSON = document.getElementById('download-json');
  const jsonBlob = new Blob([JSON.stringify(data[0])], {type: 'application/json'});
  const jsonUrl = window.URL.createObjectURL(jsonBlob);
  downloadJSON.href = jsonUrl;
  downloadJSON.download = 'model.json';

  const downloadWeights = document.getElementById('download-weights');
  const weightsBlob = new Blob([data[1]], {type: 'application/octet-stream'});
  const weightsUrl = window.URL.createObjectURL(weightsBlob);
  downloadWeights.href = weightsUrl;
  downloadWeights.download = 'weights.bin';

  const uploadJSON = document.getElementById('upload-json');
  const uploadWeights = document.getElementById('upload-weights');
  const uploadModelButton = document.getElementById('upload-model');
  console.log('uploadModelButton: ', uploadModelButton);  // DEBUG
  uploadModelButton.addEventListener('click', () => {
    console.log(uploadJSON.files);
    if (uploadJSON.files.length === 1) {
      const reader = new FileReader();  // TODO(cais): Use singleton?
      reader.onloadend = (event) => {
        console.log('model.json load end: ' + event.target.result);  // DEBUG
      };
      reader.readAsText(uploadJSON.files[0]);
    }
    console.log(uploadWeights.files);
    if (uploadWeights.files.length === 1) {
      const reader = new FileReader();  // TODO(cais): Use singleton?
      reader.onloadend = (event) => {
        const buffer = event.target.result;
        const array = new Float32Array(buffer);
        console.log('Weights load end: length: ' + array.length);  // DEBUG
        console.log('Weights load end: element 0: ' + array[0]);  // DEBUG
      };
      reader.readAsArrayBuffer(uploadWeights.files[0]);
    }
  });

  // downloadWeights.addEventListener('click', () => {
  //   console.log('Download weights!');
  //   downloadWeights.click();
  // });
}

runExportModelDemo();
