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
 * Handlers for saving and loading models, including their weights.
 */

// tslint:disable:max-line-length
import {ValueError} from '../errors';
import {ModelAndWeightsConfig} from '../types';
import {version} from '../version';
// tslint:enable:max-line-length

export type SaveModelHandler =
    (config: ModelAndWeightsConfig, weightsData: ArrayBuffer) => Promise<void>;

export type LoadModelHandler = () =>
    Promise<[ModelAndWeightsConfig, ArrayBuffer]>;

export function toLocalStorage(modelName: string): SaveModelHandler {
  if (localStorage === undefined) {
    throw new ValueError(
        'This browser environment does not support local storage.');
  }
  if (modelName == null || modelName.length === 0) {
    throw new ValueError('Must define a non-null, non-empty modelName.');
  }
  return async (config: ModelAndWeightsConfig, weightsData: ArrayBuffer) => {
    localStorage.setItem(
        modelName + '-config',
        JSON.stringify({config, tfjsLayersVersion: version}));
    localStorage.setItem(
        modelName + '-weights',
        btoa(String.fromCharCode.apply(null, new Uint8Array(weightsData))));
  };
}

export function toDownloadAnchors(
    modelJSONAnchor: HTMLAnchorElement, weightsDataAnchor: HTMLAnchorElement,
    modelJSONFileName = 'model.json', weightsDataFileName = 'weights.bin',
    triggerDownload = false) {
  if (!weightsDataFileName) {
    throw new ValueError(
        'Empty, null or undefined modelJSONFileName is not allowed.');
  }
  if (!weightsDataFileName) {
    throw new ValueError(
        'Empty, null or undefined weightsDataFileName is not allowed.');
  }
  return async (config: ModelAndWeightsConfig, weightsData: ArrayBuffer) => {
    config.weightsManifest[0].paths = [`./${weightsDataFileName}`];

    console.log('modelAndWeightsConfig = ', config);  // DEBUG
    const jsonBlob =
        new Blob([JSON.stringify(config)], {type: 'application/json'});
    const jsonUrl = window.URL.createObjectURL(jsonBlob);
    modelJSONAnchor.href = jsonUrl;
    modelJSONAnchor.download = modelJSONFileName;

    console.log('weightsData = ', weightsData);  // DEBUG
    const weightsBlob =
        new Blob([weightsData], {type: 'application/octet-stream'});
    const weightsUrl = window.URL.createObjectURL(weightsBlob);
    weightsDataAnchor.href = weightsUrl;
    weightsDataAnchor.download = weightsDataFileName;

    if (triggerDownload) {
      modelJSONAnchor.click();
      weightsDataAnchor.click();
    }
  };
}

// TODO(cais):
// export function fromLocalStorage(modelName: string): LoadModelHandler {
//   if (modelName == null || modelName.length === 0) {
//     throw new ValueError('Must define a non-null, non-empty modelName.');
//   }
//   return () => {
//     const configJSON = JSON.parse(localStorage.getItem(modelName + '-config')
//        as ModelAndWeightsConfig;
//     const base64WeightsData = localStoraget.getItem(modelName + '-weights');

//   };
// }
