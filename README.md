# Galaxy2Galaxy

Galaxy2Galaxy, or G2G for short, is a library of models, datasets, and utilities to build generative models for astronomical images, based on the Tensor2Tensor library. Similarly to T2T, the goal of this project is to accelerate research in machine
learning models applied to astronomical image processing problems.

## Usage

To generate the COSMOS 25.2 sample at native pixel scale and stamp size:

```bash
$ t2t-datagen --t2t_usr_dir=galaxy2galaxy/data_generators --problem=galsim_cosmos --data_dir=/data2/g2g/cosmos
```
This uses GalSim to draw postage stamps and save them in TFRecord format which can then be used for training.

