import math
import os
import pickle
import sys

from absl import logging
import absl.flags
from magenta.models.gansynth.lib import estimators
from magenta.models.gansynth.lib import flags as lib_flags
from magenta.models.gansynth.lib import generate_util as gu
from magenta.models.gansynth.lib import model as lib_model
from magenta.models.gansynth.lib import util
import numpy as np
import tensorflow.compat.v1 as tf

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
  "Batch size for generation."
)
absl.flags.DEFINE_integer(
  "random_z_count",
  None,
  "Number of random latent vectors to sample."
)
# absl.flags.DEFINE_string(
#   "activations_in_file",
#   None,
#   ".npy file to load activations from"
# )
# absl.flags.DEFINE_string(
#   "activations_out_file",
#   None,
#   ".npy file to save activations to"
# )
absl.flags.DEFINE_string(
  "pca_out_file",
  None,
  ".pickle file to save PCA result to"
)
absl.flags.DEFINE_string(
  "estimator",
  "fbpca",
  "PCA estimator to use"
)
absl.flags.DEFINE_integer(
  "pca_batch_size",
  None,
  "Batch size for PCA (if using an estimator with batch support)"
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

def generate_activations(model, layer, pitch, n):
  zs = model.generate_z(n)
  pitches = np.array([pitch] * n)
  #logging.info("batch_size = {}".format(FLAGS.batch_size))
  #logging.info("zs.shape = {}".format(zs.shape))
  #logging.info("pitches.shape = {}".format(pitches.shape))

  # returns a rank 4 array, e.g. layer conv1_2 has shape (random_z_count, 4, 32, 256)
  activations = model.generate_layer_outputs_from_z(
    zs,
    pitches,
    layer_names=[layer]
  )[layer]
  return activations

def generate_activations_batches(model, layer, pitch, batch_size, n):
  n_batches = math.ceil(n / batch_size)
  for i in range(n_batches):
    logging.info("generating activations batch {}/{}".format(i+1, n_batches))
    yield generate_activations(model, layer, pitch, batch_size)

def reshape_for_pca(activations, activation_shape):
  logging.info("activations.shape = {}".format(activations.shape))

  # reshape to rank 2 for PCA, preserving the first dimension and flattening the rest
  activations = activations.reshape((activations.shape[0], np.prod(activation_shape)))
      
  logging.info("reshaped activations.shape = {}".format(activations.shape))

  return activations
    
def main(unused_argv):
  absl.flags.FLAGS.alsologtostderr = True
  
  if FLAGS.seed != None:
    np.random.seed(FLAGS.seed)
    tf.random.set_random_seed(FLAGS.seed)

  if sum((int(f) for f in [FLAGS.list_layers, FLAGS.random_z_count != None])) != 1:
    logging.info("exactly one of --list_layers or --random_z_count must be specified")
    sys.exit(1)

  model = None
  if FLAGS.list_layers or FLAGS.random_z_count != None:
    # Load the model
    flags = lib_flags.Flags({
      "batch_size_schedule": [FLAGS.batch_size]
    })
    model = lib_model.Model.load_from_path(FLAGS.ckpt_dir, flags)
    
  if FLAGS.list_layers:
    for name, layer in model.fake_data_endpoints.items():
      internal_name = layer.name
      logging.info(name)
      logging.info("  name: {}".format(internal_name))
      logging.info("  shape: {}".format(layer.shape))
      logging.info("  min. random_z_count: {}".format(product(layer.shape[1:])))
    return

  assert FLAGS.random_z_count != None

  activation_shape = model.fake_data_endpoints[FLAGS.layer].shape.as_list()[1:]
  n_components = np.prod(activation_shape)
  logging.info("activation_shape = {}".format(activation_shape))
  logging.info("n_components = {}".format(n_components))

  # only applies to spca estimator but we need to provide it anyway
  sparsity = 1.0
  
  estimator = estimators.get_estimator(FLAGS.estimator, n_components, sparsity)
  
  if estimator.batch_support:
    pca_batch_size = FLAGS.pca_batch_size or n_components
    
    assert pca_batch_size >= n_components
    assert FLAGS.random_z_count % pca_batch_size == 0, "random_z_count={} is not evenly divisible by pca_batch_size={}".format(FLAGS.random_z_count, pca_batch_size)

    n = FLAGS.random_z_count
    
    for activations in generate_activations_batches(model, FLAGS.layer, FLAGS.pitch, pca_batch_size, FLAGS.random_z_count):
      activations = reshape_for_pca(activations, activation_shape)
      logging.info("fit_partial()")
      if not estimator.fit_partial(activations):
        # fit_partial() should print an error if it fails, so just exit
        sys.exit(1)

    global_mean = estimator.transformer.mean_
  else:
    activations = generate_activations(model, FLAGS.layer, FLAGS.pitch, FLAGS.random_z_count)
    activations = reshape_for_pca(activations, activation_shape)
        
    n = activations.shape[0]
    
    assert n >= n_components

    # subtract mean
    global_mean = activations.mean(axis=0, keepdims=True, dtype=np.float32)
    activations -= global_mean
  
    logging.info("running estimator")
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
    "estimator": FLAGS.estimator,
    "comp": comp,
    "stdev": stdev,
    "var_ratio": var_ratio,
    "global_mean": global_mean
  }

  logging.info("pca_dict = {}".format(pca_dict))
  
  if FLAGS.pca_out_file != None:
    logging.info("saving PCA result to {}".format(FLAGS.pca_out_file))
    with open(FLAGS.pca_out_file, "wb") as fp:
      pickle.dump(pca_dict, fp, pickle.DEFAULT_PROTOCOL)
  
def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)

if __name__ == '__main__':
  console_entry_point()
