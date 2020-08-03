import os
from absl import flags

import tensorflow as tf
import tensorflow_hub as hub

# Encoder parameter
flags.DEFINE_string("decoder_module", default=None,
                     help="Path to decoder module.")

flags.DEFINE_string("flow_module", default=None,
                     help="Path to latent flow module.")

flags.DEFINE_string("export_dir", default=None,
                    help="Path to exported module")

FLAGS = flags.FLAGS

def main(argv):
  del argv  # unused

  tf.gfile.MakeDirs(FLAGS.export_dir)

  def generative_model_fn():
    code = hub.Module(FLAGS.flow_module)
    decoder = hub.Module(FLAGS.decoder_module)
    input_info = code.get_input_info_dict()
    inputs = {k: tf.placeholder(tf.float32, shape=input_info[k].get_shape()) for k in input_info.keys()}
    hub.add_signature(inputs=inputs, outputs=decoder(code(inputs)))
    hub.attach_message("stamp_size", decoder.get_attached_message("stamp_size", tf.train.Int64List))
    hub.attach_message("pixel_size", decoder.get_attached_message("pixel_size", tf.train.FloatList))
  generative_model_spec = hub.create_module_spec(generative_model_fn)
  generative_model = hub.Module(generative_model_spec, name="flow_module")

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    generative_model.export(FLAGS.export_dir+'/generator', sess)
  print("Model exported in "+FLAGS.export_dir+'/generator')

if __name__ == "__main__":
    tf.app.run()
