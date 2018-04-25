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
  console.log(tfc.decodeWeights);  // DEBUG
  console.log(tfc.encodeWeights);  // DEBUG
  console.log(tfl);  // DEBUG
  const model =
      tfl.sequential({
          layers: [tfl.layers.dense({units: 1, inputShape: [200]})]});
  // model.save([
  //   tf.savers.toLocalStorage('myModel'),
  //   tf.savers.toDownloadAnchors(
  //       document.getElementById('download-json'),
  //       document.getElementById('download-weights'), 'model1.json',
  //       'weights1.bin')
  // ]);

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
