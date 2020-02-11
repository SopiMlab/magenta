# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module contains a registry of dataset classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from magenta.models.gansynth.lib import spectral_ops
from magenta.models.gansynth.lib import util
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import data as contrib_data
from tensorflow.contrib import lookup as contrib_lookup

Counter = collections.Counter


class BaseDataset(object):
  """A base class for reading data from disk."""

  def __init__(self, config):
    self._train_data_path = util.expand_path(config['train_data_path'])

  def provide_one_hot_labels(self, batch_size):
    """Provides one-hot labels."""
    raise NotImplementedError

  def provide_dataset(self):
    """Provides audio dataset."""
    raise NotImplementedError

  def get_pitch_counts(self):
    """Returns a dictionary {pitch value (int): count (int)}."""
    raise NotImplementedError

  def get_pitches(self, num_samples):
    """Returns pitch_counter for num_samples for given dataset."""
    all_pitches = []
    pitch_counts = self.get_pitch_counts()
    for k, v in pitch_counts.items():
      all_pitches.extend([k]*v)
    sample_pitches = np.random.choice(all_pitches, num_samples)
    pitch_counter = Counter(sample_pitches)
    return pitch_counter


class NSynthTFRecordDataset(BaseDataset):
  """A dataset for reading NSynth from a TFRecord file."""

  def __init__(self, config):
      super().__init__(config)
      self._length = 64000
      self._channels = 1
      self._dataset = None
      self._pitch_counts = None
      
  def _get_dataset_from_path(self, shuffle_and_repeat=True):
    dataset = tf.data.Dataset.list_files(self._train_data_path)
    if shuffle_and_repeat:
        dataset = dataset.apply(contrib_data.shuffle_and_repeat(buffer_size=1000))
    dataset = dataset.apply(
        contrib_data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=20, sloppy=True))
    return dataset

  def _parse_dataset(self, dataset):
    def _parse_nsynth(record):
      """Parsing function for NSynth dataset."""
      features = {
          'pitch': tf.FixedLenFeature([1], dtype=tf.int64),
          'audio': tf.FixedLenFeature([self._length], dtype=tf.float32),
          'qualities': tf.FixedLenFeature([10], dtype=tf.int64),
          'instrument_source': tf.FixedLenFeature([1], dtype=tf.int64),
          'instrument_family': tf.FixedLenFeature([1], dtype=tf.int64),
      }

      example = tf.parse_single_example(record, features)
      wave, label = example['audio'], example['pitch']
      wave = spectral_ops.crop_or_pad(wave[tf.newaxis, :, tf.newaxis],
                                      self._length,
                                      self._channels)[0]
      return wave, label, example['instrument_source']
    return dataset.map(_parse_nsynth, num_parallel_calls=4)
  
  def _count_pitches(self):
    dataset = self._get_dataset_from_path(shuffle_and_repeat=False)
    dataset = self._parse_dataset(dataset)
    dataset = dataset.map(lambda w, p, s: p[0])
    
    sess = tf.Session()
    iterator = dataset.make_initializable_iterator()
    sess.run(iterator.initializer)
    sess.run(tf.tables_initializer())
    next_element = iterator.get_next()
    
    counts_list = [0]*128
    
    try:
      i = 0
      while True:
        if i % 1000 == 0:
            print(i)
        pitch = sess.run(next_element)
        counts_list[pitch] += 1
        i += 1
    except tf.errors.OutOfRangeError:
        pass

    counts = dict(enumerate(counts_list))
        
    self._pitch_counts = counts
  
  def provide_one_hot_labels(self, batch_size):
    """Provides one hot labels."""
    pitch_counts = self.get_pitch_counts()
    pitches = sorted(pitch_counts.keys())
    counts = [pitch_counts[p] for p in pitches]
    indices = tf.reshape(
        tf.multinomial(tf.log([tf.to_float(counts)]), batch_size), [batch_size])
    one_hot_labels = tf.one_hot(indices, depth=len(pitches))
    return one_hot_labels

  def provide_dataset(self, instrument_sources=None, min_pitch=24, max_pitch=84):
    """Provides dataset (audio, labels) of nsynth."""

    pitch_counts = self.get_pitch_counts()
    pitches = sorted(pitch_counts.keys())
    label_index_table = contrib_lookup.index_table_from_tensor(
        sorted(pitches), dtype=tf.int64)

    def _add_one_hot_label(data):
      wave, label, instrument_source = data
      one_hot_label = tf.one_hot(
          label_index_table.lookup(label), depth=len(pitches))[0]
      return wave, one_hot_label, label, instrument_source
    
    dataset = self._get_dataset_from_path()
    dataset = self._parse_dataset(dataset)
    dataset = dataset.map(_add_one_hot_label)

    # Filter just specified instrument sources
    # (0=acoustic, 1=electronic, 2=synthetic)
    if instrument_sources != None:
        def _is_wanted_source(s):
            return any(map(lambda q: tf.equal(s, q)[0], instrument_sources))
        dataset = dataset.filter(lambda w, l, p, s: _is_wanted_source(s))
    # Filter just specified pitches
    dataset = dataset.filter(lambda w, l, p, s: tf.greater_equal(p, min_pitch)[0])
    dataset = dataset.filter(lambda w, l, p, s: tf.less_equal(p, max_pitch)[0])
    dataset = dataset.map(lambda w, l, p, s: (w, l))
    return dataset

  def get_pitch_counts(self):
    if self._pitch_counts == None:
      self._count_pitches()

    return self._pitch_counts        


registry = {
    'nsynth_tfrecord': NSynthTFRecordDataset,
}
