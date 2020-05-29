from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import reduce
from magenta.models.gansynth.lib import util
import tensorflow.compat.v1 as tf
from tensorflow.contrib import lookup as contrib_lookup

from dataclasses import dataclass
from typing import Callable, List

@dataclass
class ConditionDef:
  calculate_num_tokens: Callable
  get_placeholder: Callable
  get_summary_labels: Callable
  provide_labels: Callable
  compute_error: Callable
  get_label_from_record: Callable
  required_features: List
  num_tokens: int = None

def create_instrument_family_condition(config, meta):
    families_count = 11
    family_counts = [0] * families_count
    for m in meta.values():
        family_counts[m["instrument_family"]] += 1

    n_examples = len(meta)

    families_logits = list(map(lambda p: [1.0 - p, p], map(lambda fc: fc / n_examples, family_counts)))

    return ("instrument_family", ConditionDef(
        calculate_num_tokens = lambda _: families_count,
        get_placeholder = lambda batch_size, num_tokens: tf.placeholder(tf.float32, [batch_size, num_tokens]),
        get_summary_labels = lambda batch_size, num_tokens: util.make_ordered_one_hot_vectors(batch_size, num_tokens),
        provide_labels = lambda batch_size: tf.cast(tf.transpose(tf.random.categorical(tf.log(families_logits), batch_size)), tf.float32),
        compute_error = lambda labels, logits: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(labels), logits=logits)),
        get_label_from_record = lambda record: tf.one_hot(record["instrument_family"], depth=families_count)[0],
        required_features = [
          ('instrument_family', tf.FixedLenFeature([1], dtype=tf.int64))
        ]
      ))

def create_instrument_source_condition(config, meta):
    source_count = 3
    source_counts = [0] * source_count
    for m in meta.values():
        source_counts[m["instrument_source"]] += 1

    n_examples = len(meta)

    families_logits = list(map(lambda p: [1.0 - p, p], map(lambda fc: fc / n_examples, source_counts)))

    return ("instrument_source", ConditionDef(
        calculate_num_tokens = lambda _: source_count,
        get_placeholder = lambda batch_size, num_tokens: tf.placeholder(tf.float32, [batch_size, num_tokens]),
        get_summary_labels = lambda batch_size, num_tokens: util.make_ordered_one_hot_vectors(batch_size, num_tokens),
        provide_labels = lambda batch_size: tf.cast(tf.transpose(tf.random.categorical(tf.log(families_logits), batch_size)), tf.float32),
        compute_error = lambda labels, logits: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(labels), logits=logits)),
        get_label_from_record = lambda record: tf.one_hot(record["instrument_source"], depth=source_count)[0],
        required_features = [
          ('instrument_source', tf.FixedLenFeature([1], dtype=tf.int64))
        ]
      ))

def create_qualities_condition(config, meta):
    qualities_count = 10
    quality_counts = reduce(lambda qcs, m: list(map(lambda qc, q: qc + q, qcs, m["qualities"])), meta.values(), [0] * qualities_count)
    n_examples = len(meta)
    qualities_logits = list(map(lambda p: [1.0 - p, p], map(lambda qc: qc / n_examples, quality_counts)))

    return ("qualities", ConditionDef(
        calculate_num_tokens = lambda _: qualities_count,
        get_placeholder = lambda batch_size, num_tokens: tf.placeholder(tf.float32, [batch_size, num_tokens]),
        get_summary_labels = lambda batch_size, num_tokens: util.make_ordered_one_hot_vectors(batch_size, num_tokens),
        provide_labels = lambda batch_size: tf.cast(tf.transpose(tf.random.categorical(tf.log(qualities_logits), batch_size)), tf.float32),
        compute_error = lambda labels, logits: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.stop_gradient(labels), logits=logits)),
        get_label_from_record = lambda record: tf.cast(record['qualities'], tf.float32),
        required_features = [
          ('qualities', tf.FixedLenFeature([10], dtype=tf.int64))
        ]
      ))


def create_pitch_condition(config, meta):

    def get_pitch_counts():
        if meta:
            pitch_counts = {}
            for name, note in meta.items():
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

    pitch_counts = get_pitch_counts()
    pitches = sorted(pitch_counts.keys())
    label_index_table = contrib_lookup.index_table_from_tensor(
        sorted(pitches), dtype=tf.int64)

    def provide_one_hot_labels(batch_size):
        """Provides one hot labels."""
        counts = [pitch_counts[p] for p in pitches]
        indices = tf.reshape(
            tf.multinomial(tf.log([tf.to_float(counts)]), batch_size), [batch_size])
        one_hot_labels = tf.one_hot(indices, depth=len(pitches))
        return one_hot_labels

    return ("pitch", ConditionDef(
        calculate_num_tokens= lambda label: label.shape[0].value,
        get_placeholder=lambda batch_size, _: tf.one_hot(tf.placeholder(tf.int32, [batch_size]), len(pitches)),
        get_summary_labels=lambda batch_size, num_tokens: util.make_ordered_one_hot_vectors(batch_size, num_tokens),
        provide_labels=lambda batch_size: provide_one_hot_labels(batch_size),
        compute_error=lambda labels, logits: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(labels), logits=logits)),
        get_label_from_record=lambda record: tf.one_hot(label_index_table.lookup(record['pitch']), depth=len(pitches))[0],
        required_features=[
          ('pitch', tf.FixedLenFeature([1], dtype=tf.int64))
        ]
      ))