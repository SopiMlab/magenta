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
from functools import reduce
import json
import os
from magenta.models.gansynth.lib import spectral_ops
from magenta.models.gansynth.lib import util
import numpy as np
import re
import tensorflow.compat.v1 as tf
from tensorflow.contrib import data as contrib_data
from tensorflow.contrib import lookup as contrib_lookup

from magenta.models.gansynth.lib import conditions

Counter = collections.Counter


class BaseDataset(object):
  """A base class for reading data from disk."""

  def __init__(self, config):
    self._train_data_path = util.expand_path(config['train_data_path'])

  def provide_dataset(self):
    """Provides audio dataset."""
    raise NotImplementedError

  def get_conditions(self):
    raise NotImplementedError

  def is_included(self, example):
    raise NotImplementedError

  def get_pitch_counts(self):
    """Returns a dictionary {pitch value (int): count (int)}."""
    raise NotImplementedError

class NSynthGenericConditionsTFRecordDataset(BaseDataset):
  def __init__(self, config):
    super(NSynthGenericConditionsTFRecordDataset, self).__init__(config)

    self._train_meta_path = None
    if 'train_meta_path' in config and config['train_meta_path']:
      self._train_meta_path = util.expand_path(config['train_meta_path'])
    else:
      magic_meta_path = os.path.join(config['train_root_dir'], 'meta.json')
      if os.path.exists(magic_meta_path):
        self._train_meta_path = magic_meta_path

    self._instrument_sources = config['train_instrument_sources']
    self._min_pitch = config['train_min_pitch']
    self._max_pitch = config['train_max_pitch']

    self.meta = None
    with open(self._train_meta_path, "r") as meta_fp:
      self.meta = json.load(meta_fp)
      self.meta = {k: v for k, v in self.meta.items() if self.is_included(v)}

  def provide_dataset(self):
    """Provides dataset (audio, labels) of nsynth."""
    length = 64000
    channels = 1

    def _parse_nsynth(record):
      """Parsing function for NSynth dataset."""
      required_features = dict(reduce(list.__add__, [c.required_features for c in self.conditions.values()]))
      features = {
        'pitch': tf.FixedLenFeature([1], dtype=tf.int64),
        'audio': tf.FixedLenFeature([length], dtype=tf.float32),
        'instrument_source': tf.FixedLenFeature([1], dtype=tf.int64),
        **required_features
      }

      example = tf.parse_single_example(record, features)
      wave, pitch = example['audio'], example['pitch']
      wave = spectral_ops.crop_or_pad(wave[tf.newaxis, :, tf.newaxis],
                                      length,
                                      channels)[0]

      condition_labels = []

      for k, c in self.conditions.items():
        value = c.get_label_from_record(example)
        condition_labels.append((k, value))
        if c.num_tokens == None:
          c.num_tokens = c.calculate_num_tokens(value)

      condition_labels = collections.OrderedDict(condition_labels)

      return wave, condition_labels, pitch, example['instrument_source']

    dataset = self._get_dataset_from_path()
    dataset = dataset.map(_parse_nsynth, num_parallel_calls=4)

    # Filter just specified instrument sources
    def _is_wanted_source(s):
      return tf.reduce_any(list(map(lambda q: tf.equal(s, q)[0], self._instrument_sources)))

    dataset = dataset.filter(lambda w, cl, p, s: _is_wanted_source(s))
    # Filter just specified pitches
    dataset = dataset.filter(lambda w, cl, p, s: tf.greater_equal(p, self._min_pitch)[0])
    dataset = dataset.filter(lambda w, cl, p, s: tf.less_equal(p, self._max_pitch)[0])
    dataset = dataset.map(lambda w, cl, p, s: (w, cl))
    return dataset

    return one_hot_labels

  def is_included(self, example):
    return self._min_pitch <= example["pitch"] <= self._max_pitch and example["instrument_source"] in self._instrument_sources

  def get_conditions(self):
    return self.conditions

  def _get_dataset_from_path(self):
    dataset = tf.data.Dataset.list_files(self._train_data_path)
    dataset = dataset.apply(contrib_data.shuffle_and_repeat(buffer_size=1000))
    dataset = dataset.apply(
        contrib_data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=20, sloppy=True))
    return dataset

  #TODO: This could still be refactored away, only used in model generate methods to choose random pitches.
  def get_pitch_counts(self):
    if self._train_meta_path:
      with open(self._train_meta_path, "r") as meta_fp:
        meta = json.load(meta_fp)
      pitch_counts = {}
      for name, note in meta.items():
        if self.is_included(note):
          pitch = note["pitch"]
          if pitch in pitch_counts:
            pitch_counts[pitch] += 1
          else:
            pitch_counts[pitch] = 1
    else:
      pitch_counts = {
        24: 711,
        25: 720,
        26: 715,
        27: 725,
        28: 726,
        29: 723,
        30: 738,
        31: 829,
        32: 839,
        33: 840,
        34: 860,
        35: 870,
        36: 999,
        37: 1007,
        38: 1063,
        39: 1070,
        40: 1084,
        41: 1121,
        42: 1134,
        43: 1129,
        44: 1155,
        45: 1149,
        46: 1169,
        47: 1154,
        48: 1432,
        49: 1406,
        50: 1454,
        51: 1432,
        52: 1593,
        53: 1613,
        54: 1578,
        55: 1784,
        56: 1738,
        57: 1756,
        58: 1718,
        59: 1738,
        60: 1789,
        61: 1746,
        62: 1765,
        63: 1748,
        64: 1764,
        65: 1744,
        66: 1677,
        67: 1746,
        68: 1682,
        69: 1705,
        70: 1694,
        71: 1667,
        72: 1695,
        73: 1580,
        74: 1608,
        75: 1546,
        76: 1576,
        77: 1485,
        78: 1408,
        79: 1438,
        80: 1333,
        81: 1369,
        82: 1331,
        83: 1295,
        84: 1291
      }
    return pitch_counts


class NSynthTFRecordDataset(NSynthGenericConditionsTFRecordDataset):
  """A dataset for reading NSynth from a TFRecord file."""

  def __init__(self, config):
    super(NSynthTFRecordDataset, self).__init__(config)
    self.conditions = collections.OrderedDict([
      conditions.create_pitch_condition(config, self.meta)
    ])


class NSynthQualitiesTFRecordDataset(NSynthGenericConditionsTFRecordDataset):
  def __init__(self, config):
    super(NSynthQualitiesTFRecordDataset, self).__init__(config)
    
    self.conditions = collections.OrderedDict([
      conditions.create_pitch_condition(config, self.meta),
      conditions.create_qualities_condition(config, self.meta)
    ])

class NSynthInstrumentFamilyTFRecordDataset(NSynthGenericConditionsTFRecordDataset):
  def __init__(self, config):
    super(NSynthInstrumentFamilyTFRecordDataset, self).__init__(config)
    
    self.conditions = collections.OrderedDict([
      conditions.create_pitch_condition(config, self.meta),
      conditions.create_instrument_family_condition(config, self.meta)
    ])


class NSynthInstrumentFamilyOnlyTFRecordDataset(NSynthGenericConditionsTFRecordDataset):
  def __init__(self, config):
    super(NSynthInstrumentFamilyOnlyTFRecordDataset, self).__init__(config)

    self.conditions = collections.OrderedDict([
      conditions.create_instrument_family_condition(config, self.meta)
    ])

class NSynthInstrumentSourceOnlyTFRecordDataset(NSynthGenericConditionsTFRecordDataset):
  def __init__(self, config):
    super(NSynthInstrumentSourceOnlyTFRecordDataset, self).__init__(config)

    self.conditions = collections.OrderedDict([
      conditions.create_instrument_source_condition(config, self.meta)
    ])

class NSynthInstrumentSourceTFRecordDataset(NSynthGenericConditionsTFRecordDataset):
  def __init__(self, config):
    super(NSynthInstrumentSourceTFRecordDataset, self).__init__(config)

    self.conditions = collections.OrderedDict([
      conditions.create_pitch_condition(config, self.meta),
      conditions.create_instrument_source_condition(config, self.meta)
    ])

  
registry = {
    'nsynth_tfrecord': NSynthTFRecordDataset,
    'nsynth_qualities_tfrecord': NSynthQualitiesTFRecordDataset,
    'nsynth_instrument_family_tfrecord': NSynthInstrumentFamilyTFRecordDataset,
    'nsynth_instrument_family_only_tfrecord': NSynthInstrumentFamilyTFRecordDataset,
    'nsynth_instrument_source_only_tfrecord': NSynthInstrumentSourceOnlyTFRecordDataset,
    'nsynth_instrument_source_tfrecord': NSynthInstrumentSourceTFRecordDataset
}
