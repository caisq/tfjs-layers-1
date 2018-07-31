# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import shutil
import subprocess
import tempfile

import keras
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs


def _call_command(command):
  print('Calling in %s: %s' % (os.getcwd(), ' '.join(command)))
  subprocess.check_call(command)


class Tfjs2KerasExportTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    print('Preparing TensorFlow.js...')
    cls._tmp_dir = tempfile.mkdtemp()
    cwd = os.path.dirname(os.path.realpath(__file__))
    os.chdir(os.path.join(cwd, '..', '..'))
    _call_command(['yarn'])
    _call_command(['yarn', 'build'])
    _call_command(['yarn', 'link'])

    os.chdir(cwd)
    # _call_command(['yarn', 'link', '@tensorflow/tfjs-layers'])
    _call_command(['yarn'])
    _call_command(['yarn', 'build'])  # TODO(cais): Decide.
    _call_command(['node', 'dist/tfjs_save.js', cls._tmp_dir])

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls._tmp_dir)

  def _loadAndTestModel(self, model_path):
    """Load a Keras Model from artifacts generated by tensorflow.js.

    This method tests:
      - Python Keras loading of the topology JSON file saved from TensorFlow.js.
      - Python Keras loading of the model's weight values.
      - The equality of the model.predict() output between Python Keras and
        TensorFlow.js (up to a certain numeric tolerance.)

    Args:
      model_path: Path to the model JSON file.
    """
    xs_shape_path = os.path.join(
        self._tmp_dir, model_path + '.xs-shape.json')
    xs_data_path = os.path.join(
        self._tmp_dir, model_path + '.xs-data.json')
    with open(xs_shape_path, 'rt') as shape_f, open(xs_data_path, 'rt') as data_f:
      xs = np.array(
          json.load(data_f), dtype=np.float32).reshape(json.load(shape_f))

    ys_shape_path = os.path.join(
        self._tmp_dir, model_path + '.ys-shape.json')
    ys_data_path = os.path.join(
        self._tmp_dir, model_path + '.ys-data.json')
    with open(ys_shape_path, 'rt') as shape_f, open(ys_data_path, 'rt') as data_f:
      ys = np.array(
          json.load(data_f), dtype=np.float32).reshape(json.load(shape_f))

    with tf.Graph().as_default(), tf.Session():
      model_json_path = os.path.join(self._tmp_dir, model_path, 'model.json')
      print('Loading model from path %s' % model_json_path)
      model = tfjs.converters.load_keras_model(model_json_path)
      ys_new = model.predict(xs)
      print(ys - ys_new)  # DEBUG
      self.assertAllClose(ys, ys_new)

  def testMLP(self):
    self._loadAndTestModel('mlp')

  def testCNN(self):
    self._loadAndTestModel('cnn')

  def testDepthwiseCNN(self):
    self._loadAndTestModel('depthwise_cnn')

  def testSimpleRNN(self):
    self._loadAndTestModel('simple_rnn')

  def testGRU(self):
    self._loadAndTestModel('gru')

  def testBidirectionalLSTM(self):
    self._loadAndTestModel('bidirectional_lstm')

  def testTimeDistributedLSTM(self):
    self._loadAndTestModel('time_distributed_lstm')

  def testOneDimensional(self):
    self._loadAndTestModel('one_dimensional')

  def testFunctionalMerge(self):
    self._loadAndTestModel('functional_merge.json')


if __name__ == '__main__':
  tf.test.main()

