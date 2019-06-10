# Galaxy2Galaxy

Galaxy2Galaxy, or G2G for short, is a library of models, datasets, and utilities to build generative models for astronomical images, based on the Tensor2Tensor library. Similarly to T2T, the goal of this project is to accelerate research in machine
learning models applied to astronomical image processing problems.

## Usage

To generate the COSMOS 25.2 sample at native pixel scale and stamp size:

```bash
$ t2t-datagen --t2t_usr_dir=galaxy2galaxy/data_generators --problem=galsim_cosmos32 --data_dir=/data2/g2g/cosmos32
```
This uses GalSim to draw postage stamps and save them in TFRecord format which can then be used for training.

To  train an autoencoder with this data:
```bash
$ t2t-trainer --t2t_usr_dir=galaxy2galaxy/ --data_dir=/data2/g2g/cosmos32   --output_dir=/data2/g2g/train/cosmos32   --problem=img2img_galsim_cosmos32   --model=continuous_autoencoder_basic  --train_steps=2000   --eval_steps=100 --hparams_set=continuous_autoencoder_basic
```
