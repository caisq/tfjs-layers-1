/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import * as tfl from './index';
import {describeMathCPU} from './utils/test_utils';

describeMathCPU('Model.summary', () => {
  it('Sequential model: one layer', () => {
    const model = tfl.sequential(
        {layers: [tfl.layers.dense({units: 3, inputShape: [10]})]});
    model.summary();
  });
});
