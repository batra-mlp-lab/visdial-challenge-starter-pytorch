Visual Dialog Challenge Starter Code
====================================

PyTorch starter code for the [Visual Dialog Challenge][1].

  * [Setup and Dependencies](#setup-and-dependencies)
  * [Preprocessing VisDial](#preprocessing-visdial)
  * [Extracting Image Features](#extracting-image-features)
  * [Download Preprocessed Data](#download-preprocessed-data)
  * [Training](#training)
  * [Evaluation](#evaluation)
  * [Generate Submission](#generate-submission)
  * [Pretrained Checkpoint](#pretrained-checkpoint)
  * [Acknowledgements](#acknowledgements)
  * [License](#license)


Setup and Dependencies
----------------------

Our code is implemented in PyTorch (v0.3.0 with CUDA). To setup, do the following:

If you do not have any Anaconda or Miniconda distribution, head over to their [downloads' site][2] before proceeding further.

Clone the repository and create an environment.

```sh
git clone https://www.github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
conda env create -f env.yml
```
This creates an environment named `visdial-chal` with all the dependencies installed.

If you wish to extract your own image features, you require a Torch distribution. Skip everything in this subsection from here if you will not extract your own features.

```sh
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
TORCH_LUA_VERSION=LUA51 ./install.sh
```

Additionally, image feature extraction code uses [torch-hdf5][3], [torch/image][4] and [torch/loadcaffe][5]. After Torch is installed, these can be installed/updated using:

```sh
luarocks install image
luarocks install loadcaffe
```

Installation instructions for torch-hdf5 are given [here][6].
Optionally, these packages are required for GPU acceleration:

```sh
luarocks install cutorch
luarocks install cudnn
luarocks install cunn
```

**Note:** Since Torch is in maintenance mode now, it requires CUDNN v5.1 or lower. Install it separately and set `$CUDNN_PATH` environment variable to the binary (shared object) file.


Download Preprocessed Data
--------------------------

We provide preprocessed files for VisDial v1.0 (tokenized captions, questions, answers, image indices, vocabulary mappings and image features extracted by pretrained CNN). If you wish to preprocess data or extract your own features, skip this step.

Extracted features for v1.0 train, val and test are available for download [here][7].

* `visdial_data_train.h5`: Tokenized captions, questions, answers, image indices, for training on `train`
* `visdial_params_train.json`: Vocabulary mappings and COCO image ids for training on `train`
* `data_img_vgg16_relu7_train.h5`: VGG16 `relu7` image features for training on `train`
* `data_img_vgg16_pool5_train.h5`: VGG16 `pool5` image features for training on `train`
* `visdial_data_trainval.h5`: Tokenized captions, questions, answers, image indices, for training on `train`+`val`
* `visdial_params_trainval.json`: Vocabulary mappings and COCO image ids for training on `train`+`val`
* `data_img_vgg16_relu7_trainval.h5`: VGG16 `relu7` image features for training on `train`+`val`
* `data_img_vgg16_pool5_trainval.h5`: VGG16 `pool5` image features for training on `train`+`val`

Download these files to `data` directory. If you are downloaded just one file each for `visdial_data*.h5`, `visdial_params*.json`, `data_img*.h5`, it would be convenient to rename them and remove everything represented by asterisk. These names are used in default arguments of train and evaluate scripts.


Preprocessing VisDial
---------------------

Download all the images required for VisDial v1.0. Create an empty directory anywhere and place four subdirectories with the downloaded images, named:
  - [`train2014`][8] and [`val2014`][9] from COCO dataset, used by `train` split.
  - `VisualDialog_val2018` and `VisualDialog_test2018` - can be downloaded from [here][10].

This shall be referred as the image root directory.

```sh
cd data
python prepro.py -download -image_root /path/to/images
cd ..
```

This script will generate the files `data/visdial_data.h5` (contains tokenized captions, questions, answers, image indices) and `data/visdial_params.json` (contains vocabulary mappings and COCO image ids).


Extracting Image Features
-------------------------

Since we don't finetune the CNN, training is significantly faster if image features are pre-extracted. Currently this repository provides support for extraction from VGG-16 and ResNets. We use image features from [VGG-16][11].

To extract image features using VGG-16, run the following:

```sh
sh data/download_model.sh vgg 16
cd data

th prepro_img_vgg16.lua -imageRoot /path/to/images -gpuid 0

```
Similary, to extract features using [ResNet][12], run:

```sh
sh data/download_model.sh resnet 200
cd data
th prepro_img_resnet.lua -imageRoot /path/to/images -cnnModel /path/to/t7/model -gpuid 0
```

Running either of these should generate `data/data_img.h5` containing features for `train`, `val` and `test` splits corresponding to VisDial v1.0.


Training
--------

This codebase supports discriminative decoding only; read more [here][16]. For reference, we have Late Fusion Encoder from the Visual Dialog paper.

Training works with default arguments by:
```sh
python train.py -encoder lf-ques-im-hist -decoder disc -gpuid 0  # other args
```

The script has all the default arguments, so it works without specifying any arguments. Execute the script with `-h` to see a list of available arguments which can be changed as per need (such as learning rate, epochs, batch size, etc).

To extend this starter code, add your own encoder/decoder modules into their respective directories and include their names as choices in command line arguments of `train.py`.

We have an `-overfit` flag, which can be useful for rapid debugging. It takes a batch of 5 examples and overfits the model on them.


Evaluation
----------

Evaluation of a trained model checkpoint can be done as follows:

```sh
python evaluate.py -split val -load_path /path/to/pth/checkpoint -use_gt
```

To evaluate on metrics from the [Visual Dialog paper][13] (Mean reciprocal rank, R@{1, 5, 10}, Mean rank), use the `-use_gt` flag. Since the `test` split has no ground truth, `-split test` won't work here.

**Note:** The metrics reported here would be the same as those reported through EvalAI by making a submission in `val` phase.


Generate Submission
-------------------

To save predictions in a format submittable to the evaluation server on EvalAI, run the evaluation script (without using the `-use_gt` flag).

To generate a submission file for `val` phase:
```sh
python evaluate.py -split val -load_path /path/to/pth/checkpoint -save_ranks -save_path /path/to/submission/json
```

To generate a submission file for `test-std` or `test-challenge` phase, replace `-split val` with `-split test`.


Pretrained Checkpoint
---------------------

Pretrained checkpoint of Late Fusion Encoder - Discriminative Decoder model is available [here][17].

Performance on `v1.0` val (trained on `v1.0` train):

|  R@1   |  R@5   |  R@10  | MeanR  |  MRR   |
| ------ | ------ | ------ | ------ | ------ |
| 0.4298 | 0.7464 | 0.8491 | 5.4874 | 0.5757 |


Acknowledgements
----------------

* This starter code began as a fork of [batra-mlp-lab/visdial-rl][14]. We thank the developers for doing most of the heavy-lifting.
* The Lua-torch codebase of Visual Dialog, at [batra-mlp-lab/visdial][15], served as an important reference while developing this codebase. 


License
=======

BSD


[1]: https://visualdialog.org/challenge/2018
[2]: https://conda.io/docs/user-guide/install/download.html
[3]: https://www.github.com/deepmind/torch-hdf5
[4]: https://www.github.com/torch/image
[5]: https://www.github.com/szagoruyko/loadcaffe
[6]: https://github.com/deepmind/torch-hdf5/blob/master/doc/usage.md
[7]: https://computing.ece.vt.edu/~abhshkdz/visdial/data/v1.0/
[8]: http://images.cocodataset.org/zips/train2014.zip
[9]: http://images.cocodataset.org/zips/val2014.zip
[10]: https://visualdialog.org/data
[11]: http://www.robots.ox.ac.uk/~vgg/research/very_deep/
[12]: https://github.com/facebook/fb.resnet.torch/tree/master/pretrained
[13]: https://arxiv.org/abs/1611.08669
[14]: https://www.github.com/batra-mlp-lab/visdial-rl
[15]: https://www.github.com/batra-mlp-lab/visdial
[16]: https://visualdialog.org/challenge/2018#faq
[17]: https://www.dropbox.com/s/w40h26rhsqpbmjx/lf-ques-im-hist-vgg16-train.pth
