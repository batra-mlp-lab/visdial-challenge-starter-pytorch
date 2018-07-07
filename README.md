Visual Dialog Challenge Starter Code
====================================

Pytorch starter code for [Visual Dialog Challenge][1].


Table of Contents
=================

  * [Setup and Dependencies](#setup-and-dependencies)
  * [Preprocessing VisDial](#preprocessing-visdial)
  * [Extracting Image Features](#extracting-image-features)
  * [Download Preprocessed Data](#download-preprocessed-data)
  * [Training](#training)
  * [Evaluation](#evaluation)
  * [Generate Submission](#submission)
  * [Acknowledgements](#acknowledgements)
  * [License](#license)


Setup and Dependencies
======================

Our code is implemented in PyTorch (v0.3.0 with CUDA). To setup, do the following:

If you do not have any Anaconda or Miniconda distribution, head over to their [downloads site][11] before proceeding further.

Clone the repository and create an environment.

```sh
git clone https://www.github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
conda env create -f env.yml
```
This creates an environment named `visdial-chal` with all the dependencies installed.

We provide image features extracted from `relu_7` layer of VGG16 to use directly with training / evaluation. If you wish to extract your own image features, you require a [Torch][4] distribution. Skip everything in this subsection from here if you will not extract your own features.

```sh
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
TORCH_LUA_VERSION=LUA51 ./install.sh
```

Additionally, image feature extraction code uses [torch-hdf5][5], [torch/image][6] and [torch/loadcaffe][7]. After Torch is installed, these can be installed/updated using:

```sh
luarocks install image
luarocks install loadcaffe
```

Installation instructions for torch-hdf5 are given [here][12].
Optionally, these packages are required for GPU acceleration:

```sh
luarocks install cutorch
luarocks install cudnn
luarocks install cunn
```

Note that since Torch is in maintainance mode now, it requires CUDNN v5.1 or lower. Install it separately and set `$CUDNN_PATH` environment variable to the binary (shared object) file.


Download Preprocessed Data
==========================

We provide the preprocessed VisDial v1.0 dataset (tokenized captions, questions, answers, image indices, vocabulary mappings and image features extracted by pretrained CNN). If you wish to preprocess data or extract your own features, skip this step.

Extracted features for v1.0 train, val and test are available for download [here][18].

* `visdial_data_train.h5`: Tokenized captions, questions, answers, image indices, for training on `train`
* `visdial_params_train.json`: Vocabulary mappings and COCO image ids for training on `train`
* `data_img_vgg16_relu7_train.h5`: VGG16 `relu7` image features for training on `train`
* `data_img_vgg16_pool5_train.h5`: VGG16 `pool5` image features for training on `train`
* `visdial_data_trainval.h5`: Tokenized captions, questions, answers, image indices, for training on `train`+`val`
* `visdial_params_trainval.json`: Vocabulary mappings and COCO image ids for training on `train`+`val`
* `data_img_vgg16_relu7_trainval.h5`: VGG16 `relu7` image features for training on `train`+`val`
* `data_img_vgg16_pool5_trainval.h5`: VGG16 `pool5` image features for training on `train`+`val`


Preprocessing VisDial
=====================

Download all the image files required for VisDial v1.0. In the root directory, there should be four subdirectories, named:
  - [`train2014`][13] and [`val2014`][14] from COCO dataset, used by `train` split.
  - `VisualDialog_val2018` and `VisualDialog_test2018` - can be downloaded from [here][15].

```sh
cd data
python prepro.py -download -image_root /path/to/images
cd ..
```

This script will generate the files `data/visdial_data.h5` (contains tokenized captions, questions, answers, image indices) and `data/visdial_params.json` (contains vocabulary mappings and COCO image ids).


Extracting Image Features
=========================

Since we don't finetune the CNN, training is significantly faster if image features are pre-extracted. Currently this repository provides support for extraction from VGG-16 and ResNets. We use image features from [VGG-16][28].

To extract image features using VGG-16, run the following:

```sh
sh data/download_model.sh vgg 16
cd data

th prepro_img_vgg.lua -imageRoot /path/to/images -gpuid 0

```
Similary, to extract features using [ResNet](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained), run:

```sh
sh data/download_model.sh resnet 200
cd data
th prepro_img_resnet.lua -imageRoot /path/to/images -cnnModel /path/to/t7/model -gpuid 0
```

Running either of these should generate `data/data_img.h5` containing features for `train`, `val` and `test` splits corresponding to VisDial v1.0.


Training
========

The competition is only meant for discriminative models, so only discriminative decoder is supported here. For reference, we have Late Fusion Encoder from the Visual Dialog paper.

Training works with default arguments by:
```sh
python train.py -num_epochs 20 -batch_size 16 -gpuid 0  # other args
```

The script has all the default arguments, so it works without specifying any arguments. Execute the script with `-h` to see a list of available arguments which can be changed as per need (such as learning rate, epochs, batch size, etc).

We have an `-overfit` flag, which can be useful for rapid execution during debugging. It takes a batch of 5 examples and overfits the model on them.


Evaluation
==========

Evaluation of a trained model checkpoint can be done as follows:

```sh
python evaluate.py -split val -load_path /path/to/pth/checkpoint -use_gt
```

The `-use_gt` argument gives a signal to use the ground truth from split and hence, is necessary to calculate the evaluation metrics from Visual Dialog paper (Recall@ 1, 5, 10, Mean Rank, Mean Reciprocal Rank). Since the `test` split has no ground truth, `-split test` won't work here.

**Note:** The metrics reported here would be the same as those reported through EvalAI by making a submission in `val` phase.


Generate Submission
===================

To generate the submission, simply run the evaluation script with one of out trained model checkpoints. The argument `-use_gt` should **not** be used here, to generate ranks based on the score of decoder. 

To generate a submission for `val` phase:
```sh
python evaluate.py -split val -load_path /path/to/pth/checkpoint -save_ranks -save_path /path/to/submission/json
```

To generate a submission for `test-std` or `test-challenge` phase, replace `split val` with `-split test`.


Acknowledgements
================

This starter code began as a fork of [batra-mlp-lab/visdial-rl][16]. We thank the developers for doing most of the heavy-lifting. The Lua-torch codebase of Visual Dialog, at [batra-mlp-lab/visdial][17] served as an important reference while developing this codebase. 


License
=======

BSD


[1]: https://visualdialog.org/challenge/2018
[2]: https://www.python.org/downloads/release/python-365/
[3]: https://pytorch.org/
[4]: http://torch.ch/
[5]: https://www.github.com/deepmind/torch-hdf5
[6]: https://www.github.com/torch/image
[7]: https://www.github.com/szagoruyko/loadcaffe
[8]: https://www.github.com/torch/cutorch
[9]: https://www.github.com/soumith/cudnn.torch
[10]: https://www.github.com/torch/cunn
[11]: https://conda.io/docs/user-guide/install/download.html
[12]: https://github.com/deepmind/torch-hdf5/blob/master/doc/usage.md
[13]: http://images.cocodataset.org/zips/train2014.zip
[14]: http://images.cocodataset.org/zips/val2014.zip
[15]: https://visualdialog.org/data
[16]: https://www.github.com/batra-mlp-lab/visdial-rl
[17]: https://www.github.com/batra-mlp-lab/visdial
[18]: https://computing.ece.vt.edu/~abhshkdz/visdial/data/v1.0/
