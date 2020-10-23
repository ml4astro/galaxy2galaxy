from galaxy2galaxy import problems
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
Modes = tf.estimator.ModeKeys

candels = problems.problem('attrs2img_candels_goods64_euclid')
dset = candels.dataset(Modes.TRAIN, data_dir='/data57/hbretonniere/datasets/candels_for_g2g/test_candels')
batch = dset.batch(1)

print("data set :", dset, "\n \n")

print("batch : ", batch, "\n \n")


it = batch.make_one_shot_iterator().get_next()
sess = tf.Session()
batch_1 = sess.run(it)