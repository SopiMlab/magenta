from functools import reduce
import os
import pickle
import sys

from absl import logging
import absl.flags
from magenta.models.gansynth.lib import flags as lib_flags
from magenta.models.gansynth.lib import generate_util as gu
from magenta.models.gansynth.lib import model as lib_model
from magenta.models.gansynth.lib import util
import numpy as np
import tensorflow.compat.v1 as tf

import estimators

absl.flags.DEFINE_string('ckpt_dir',
                         '/tmp/gansynth/acoustic_only',
                         'Path to the base directory of pretrained checkpoints.'
                         'The base directory should contain many '
                         '"stage_000*" subdirectories.')
absl.flags.DEFINE_boolean(
  "list_layers",
  False,
  "List GANSynth layer names and shapes."
)
absl.flags.DEFINE_string(
  "layer",
  "conv0",
  "Name of GANSynth layer to operate on."
)
absl.flags.DEFINE_integer(
  "batch_size",
  8,
  "Batch size for generation.")
absl.flags.DEFINE_integer(
  "random_z_count",
  None,
  "Number of random latent vectors to sample."
)
absl.flags.DEFINE_string(
  "activations_in_file",
  None,
  ".npy file to load activations from"
)
absl.flags.DEFINE_string(
  "activations_out_file",
  None,
  ".npy file to save activations to"
)
absl.flags.DEFINE_string(
  "pca_out_file",
  None,
  ".pickle file to save PCA result to"
)
absl.flags.DEFINE_integer(
  "pitch",
  32,
  "Note pitch."
)
absl.flags.DEFINE_integer(
  "seed",
  None,
  "Random seed."
)

FLAGS = absl.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

product = lambda xs: reduce(lambda a, b: a*b, xs, 1)
  
def main(unused_argv):
  absl.flags.FLAGS.alsologtostderr = True

  if FLAGS.seed != None:
    np.random.seed(FLAGS.seed)
    tf.random.set_random_seed(FLAGS.seed)

  if sum((int(f) for f in [FLAGS.list_layers, FLAGS.random_z_count != None, FLAGS.activations_in_file != None])) != 1:
    logging.info("exactly one of --list_layers, --random_z_count or --activations_in_file must be specified")
    sys.exit(1)

  model = None
  if FLAGS.list_layers or FLAGS.random_z_count != None:
    # Load the model
    flags = lib_flags.Flags({"eval_batch_size": FLAGS.batch_size})
    model = lib_model.Model.load_from_path(FLAGS.ckpt_dir, flags)

  if FLAGS.list_layers:
    for name, layer in model.fake_data_endpoints.items():
      internal_name = layer.name
      logging.info(name)
      logging.info("  name: {}".format(internal_name))
      logging.info("  shape: {}".format(layer.shape))
      logging.info("  min. random_z_count: {}".format(product(layer.shape[1:])))
    return
  elif FLAGS.random_z_count != None:
    zs = model.generate_z(FLAGS.random_z_count)
    pitches = np.array([FLAGS.pitch] * FLAGS.random_z_count)
    logging.info("batch_size = {}".format(FLAGS.batch_size))
    logging.info("zs.shape = {}".format(zs.shape))
    logging.info("pitches.shape = {}".format(pitches.shape))

    # returns a rank 4 array, e.g. layer conv1_2 has shape (random_z_count, 4, 32, 256)
    activations = model.generate_layer_outputs_from_z(
      zs,
      pitches,
      layer_names=[FLAGS.layer]
    )[FLAGS.layer]
  elif FLAGS.activations_in_file != None:
    activations = np.load(FLAGS.activations_in_file)

  logging.info("activations.shape = {}".format(activations.shape))
  
  activation_shape = activations.shape[1:]
  
  # reshape to rank 2 for PCA, preserving the first dimension and flattening the rest
  activations = activations.reshape((
    activations.shape[0],
    product(activations.shape[1:])
  ))
  
  logging.info("reshaped activations.shape = {}".format(activations.shape))

  n, n_components = activations.shape

  assert n >= n_components

  # subtract mean
  global_mean = activations.mean(axis=0, keepdims=True, dtype=np.float32)
  activations -= global_mean
  
  logging.info("running estimator")
  estimator = estimators.FacebookPCAEstimator(n_components)
  estimator.fit(activations)

  logging.info("getting components")
  comp, stdev, var_ratio = estimator.get_components()

  # normalize
  comp /= np.linalg.norm(comp, axis=-1, keepdims=True)

  # inflate
  comp = comp.reshape(-1, *activation_shape)
  global_mean = global_mean.reshape(activation_shape)
  
  pca_dict = {
    "layer": FLAGS.layer,
    "comp": comp,
    "stdev": stdev,
    "var_ratio": var_ratio
  }

  logging.info("pca_dict = {}".format(pca_dict))
  
  if FLAGS.pca_out_file != None:
    logging.info("saving PCA result to {}".format(FLAGS.pca_out_file))
    with open(FLAGS.pca_out_file, "wb") as fp:
      pickle.dump(pca_dict, fp, pickle.HIGHEST_PROTOCOL)
  
def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
