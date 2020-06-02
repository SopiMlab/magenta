

import os

import absl.flags
from magenta.models.gansynth.lib import flags as lib_flags
from magenta.models.gansynth.lib import generate_util as gu
from magenta.models.gansynth.lib import model as lib_model
from magenta.models.gansynth.lib import util
import tensorflow.compat.v1 as tf


absl.flags.DEFINE_string('ckpt_dir',
                         '/tmp/gansynth/acoustic_only',
                         'Path to the base directory of pretrained checkpoints.'
                         'The base directory should contain many '
                         '"stage_000*" subdirectories.')

FLAGS = absl.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
  absl.flags.FLAGS.alsologtostderr = True

  # Load the model
  flags = lib_flags.Flags()
  model = lib_model.Model.load_from_path(FLAGS.ckpt_dir, flags)
  # Otherwise, just generate a batch of random sounds
  results = model.generate_layer_outputs(1)
  print(results)



def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
