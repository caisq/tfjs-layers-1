/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {Model} from '..';
import {Layer, Node} from '../engine/topology';
import {countParamsInWeights} from './generic_utils';

/**
 * Print the summary of a Model object.
 *
 * @param model tf.Model instance.
 * @param lineLength Total length of printed lines. Set this to adapt to the
 *   display to different terminal or console sizes.
 * @param positions Relative or absolute positions of log eleemnts in each
 *   line.
 *   If not provided, defaults to `[]` [0.45, 0.85, 1] for sequential-like
 *   models and `[0.33, 0.55, 0.67, 1]` for non-sequential like models.
 * @param printFn Print function to use.
 *   It will be called on each line of the summary. You can provide a custom
 *   function in order to capture the string summary. Defaults to `console.log`.
 * @returns The lines as an `Array` of `string`s.
 */
export function printSummary(
    model: Model, lineLength?: number, positions?: number[],
    // tslint:disable-next-line:no-any
    printFn: (message?: any, ...optionalParams: any[]) => void =
        console.log): string[] {
  const lines: string[] = [];
  const sequentialLike = isModelSequentialLike(model);

  // Header names for different log elements.
  const toDisplay: string[] = ['Layer (type)', 'Output shape', 'Param #'];
  if (sequentialLike) {
    lineLength = lineLength || 65;
    positions = positions || [0.45, 0.85, 1];
  } else {
    lineLength = lineLength || 98;
    positions = positions || [0.33, 0.55, 0.67, 1];
    // Header names for different log elements.
  }
  if (positions[positions.length - 1] <= 1) {
    positions = positions.map(p => Math.floor(lineLength * p));
  }

  let relevantNodes: Node[];
  if (!sequentialLike) {
    toDisplay.push('Connected to');
    relevantNodes = [];
    for (const depth in model.nodesByDepth) {
      relevantNodes.push(...model.nodesByDepth[depth]);
    }
  }

  function printString(row: string) {
    printFn(row);
    lines.push(row);
  }

  printString('_'.repeat(lineLength));
  lines.push(printRow(toDisplay, positions, printFn));
  printString('='.repeat(lineLength));

  const layers = model.layers;
  for (let i = 0; i < layers.length; ++i) {
    if (sequentialLike) {
      lines.push(printLayerSummary(layers[i], positions));
    } else {
      // TODO(cais): DO NOT SUBMIT.
    }

    printString((i === layers.length - 1 ? '=' : '_').repeat(lineLength));
  }

  // tslint:disable-next-line:no-any
  (model as any).checkTrainableWeightsConsistency();
  let trainableCount: number;
  // tslint:disable:no-any
  if ((model as any).collectedTrainableWeights != null) {
    trainableCount =
        countParamsInWeights((model as any).collectedTrainableWeights);
  } else {
    trainableCount = countParamsInWeights(model.trainableWeights);
  }
  // tslint:enable:no-any
  const nonTrainableCount = countParamsInWeights(model.nonTrainableWeights);

  printString(`Total params: ${trainableCount + nonTrainableCount}`);
  printString(`Trainable params: ${trainableCount}`);
  printString(`Non-trainable params: ${nonTrainableCount}`);
  printString('_'.repeat(lineLength));

  return lines;
}

function isModelSequentialLike(model: Model): boolean {
  let sequentialLike = true;
  const nodesByDepth: Node[][] = [];
  const nodes: Node[] = [];
  for (const depth in model.nodesByDepth) {
    nodesByDepth.push(model.nodesByDepth[depth]);
  }
  for (const depthNodes of nodesByDepth) {
    if (depthNodes.length > 1 ||
        depthNodes.length === 1 && depthNodes[0].inboundLayers.length > 1) {
      sequentialLike = false;
      break;
    }
    nodes.push(...depthNodes);
  }
  if (sequentialLike) {
    // Search for shared layers.
    for (const layer of model.layers) {
      let flag = false;
      for (const node of layer.inboundNodes) {
        if (nodes.indexOf(node) !== -1) {
          if (flag) {
            sequentialLike = false;
            break;
          } else {
            flag = true;
          }
        }
      }
      if (!sequentialLike) {
        break;
      }
    }
  }
  return sequentialLike;
}

function printRow(
    fields: string[], positions: number[],
    // tslint:disable-next-line:no-any
    printFn: (message?: any, ...optionalParams: any[]) => void =
        console.log): string {
  let line = '';
  for (let i = 0; i < fields.length; ++i) {
    if (i > 0) {
      line = line.slice(0, line.length - 1) + ' ';
    }
    line += fields[i];
    line = line.slice(0, positions[i]);
    line += ' '.repeat(positions[i] - line.length);
  }
  printFn(line);
  return line;
}

/**
 * Prints a summary for a single Layer.
 *
 * @param layer: Layer instance to print.
 */
function printLayerSummary(layer: Layer, positions: number[]): string {
  let outputShape: string;
  try {
    outputShape = JSON.stringify(layer.outputShape);
  } catch (err) {
    outputShape = 'multiple';
  }

  const name = layer.name;
  const className = layer.getClassName();
  const fields: string[] =
      [`${name} (${className})`, outputShape, layer.countParams().toString()];
  return printRow(fields, positions);
}
