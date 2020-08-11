# Galaxy2Galaxy [![Build Status](https://travis-ci.org/ml4astro/galaxy2galaxy.svg?branch=master)](https://travis-ci.org/ml4astro/galaxy2galaxy) [![Documentation Status](https://readthedocs.org/projects/galaxy2galaxy/badge/?version=latest)](https://galaxy2galaxy.readthedocs.io/en/latest/?badge=latest) [![Join the chat at https://gitter.im/ml4astro/galaxy2galaxy](https://badges.gitter.im/ml4astro/galaxy2galaxy.svg)](https://gitter.im/ml4astro/galaxy2galaxy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) 

Galaxy2Galaxy, or G2G for short, is a library of models, datasets, and utilities to build generative models for astronomical images, based on the [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) library. Similarly to T2T, the goal of this project is to accelerate research in machine
learning models applied to astronomical image processing problems.

Current features:
  - Datasets:
    - Framework for building image datasets using GalSim
    - Tools for building an image dataset from HSC Public data release
  - Models:
    - Variational Auto-Encoders
    - Self-Attention GAN
    - PixelCNN
    - Normalizing Flows
    
Note that **G2G is still under development**, additional feature/dataset suggestions are most welcome, all contributions are most welcome. Don't hesitate to join the [gitter](https://gitter.im/ml4astro/galaxy2galaxy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) room for any questions!

## Install

We recommend users create a conda environment before installing galaxy2galaxy. This makes installing tensorflow and galsim very easy:
```
$ conda install tensorflow-gpu==1.15
$ conda install -c conda-forge galsim
```
G2G can then easily be installed using pip inside the environment:
```
$ pip install git+https://github.com/ml4astro/pixel-cnn.git 
$ pip install git+https://github.com/ml4astro/GalFlow.git
$ pip install galaxy2galaxy
```

## Usage

To generate the COSMOS 25.2 sample at native pixel scale and stamp size:

```bash
$ g2g-datagen --problem=img2img_cosmos --data_dir=data/img2img_cosmos
```
This uses GalSim to draw postage stamps and save them in TFRecord format which can then be used for training.  **This assumes that you have downloaded the GalSim COSMOS sample**, if that's not the case, you can dowload it with: `galsim_download_cosmos -s 25.2`

To  train an autoencoder with this data:
```bash
$ g2g-trainer --data_dir=data/img2img_cosmos --output_dir=training/cosmos_ae   --problem=img2img_cosmos --model=continuous_autoencoder_basic  --train_steps=2000  --eval_steps=100 --hparams_set=continuous_autoencoder_basic
```

## Publications making use of Galaxy2Galaxy

  - [*Deep Generative Models for Galaxy Image Simulations*, Lanusse, Mandelbaum, Ravanbakhsh, Li, Freeman, and Poczos (2020)](https://arxiv.org/abs/2008.03833)  
   [![github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/McWilliamsCenter/deep_galaxy_models) [![arXiv:2008.03833](https://img.shields.io/badge/astro--ph.IM-arXiv%3A2008.03833-B31B1B.svg)](https://arxiv.org/abs/2008.03833)
