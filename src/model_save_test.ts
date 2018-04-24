/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {IOHandler, ModelArtifacts, SaveResult} from '@tensorflow/tfjs-core';

import {Dense} from './layers/core';
import {Sequential} from './models';
import {describeMathCPUAndGPU} from './utils/test_utils';

describeMathCPUAndGPU('Model.save', () => {
  class IOHandlerForTest implements IOHandler {
    savedArtifacts: ModelArtifacts;

    constructor() {}

    async save(modelArtifacts: ModelArtifacts): Promise<SaveResult> {
      this.savedArtifacts = modelArtifacts;
      return {success: true};
    }
  }

  it('Save', async done => {  // TODO(cais): Rename test title.
    const model = new Sequential();
    model.add(new Dense({units: 3, inputShape: [5]}));
    const handler = new IOHandlerForTest();
    console.log(handler);  // DEBUG

    model.save(handler)
        .then(saveResult => {
          console.log(`saveResult = ${JSON.stringify(saveResult)}`);  // DEBUG
          expect(saveResult.success).toEqual(true);
          console.log(
              JSON.stringify(handler.savedArtifacts.modelTopology));  // DEBUG
          console.log(
              JSON.stringify(handler.savedArtifacts.weightSpecs));  // DEBUG
          console.log(handler.savedArtifacts.weightData);           // DEBUG
          done();
        })
        .catch(err => {
          console.error(err.stack);
        });
  });
});
