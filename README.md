Visual Dialog Challenge Starter Code
====================================

PyTorch starter code for the [Visual Dialog Challenge 2019][1].

  * [Setup and Dependencies](#setup-and-dependencies)
  * [Download Data](#download-data)
  * [Training](#training)
  * [Evaluation](#evaluation)
  * [Pretrained Checkpoint](#pretrained-checkpoint)
  * [Acknowledgements](#acknowledgements)

If you use this code in your research, please consider citing:

```text
@misc{desai2018visdialch,
  author =       {Karan Desai and Abhishek Das and Dhruv Batra and Devi Parikh},
  title =        {Visual Dialog Challenge Starter Code},
  howpublished = {\url{https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch}},
  year =         {2018}
}
```

[![DOI](https://zenodo.org/badge/140210239.svg)](https://zenodo.org/badge/latestdoi/140210239)


What's new with `v2019`?
------------------------

If you are a returning user (from Visual Dialog Challenge 2018), here are some key highlights about our offerings in `v2019` of this starter code:

1. _Almost_ a complete rewrite of `v2018`, which increased speed, readability, modularity and extensibility.
2. Multi-GPU support - try out specifying GPU ids to train/evaluate scripts as: `--gpu-ids 0 1 2 3`
3. Docker support - we provide a Dockerfile which can help you set up all the dependencies with ease.
4. Stronger baseline - our Late Fusion Encoder is equipped with [Bottom-up Top-Down attention][6]. We also provide pre-extracted image features (links below).
5. Minimal pre-processed data - no requirement to download tens of pre-processed data files anymore (were typically referred as `visdial_data.h5` and `visdial_params.json`).


Setup and Dependencies
----------------------

This starter code is implemented using PyTorch v1.0, and provides out of the box support with CUDA 9 and CuDNN 7.
There are two recommended ways to set up this codebase: Anaconda or Miniconda, and Docker.

### Anaconda or Miniconda

1. Install Anaconda or Miniconda distribution based on Python3+ from their [downloads' site][2].
2. Clone this repository and create an environment:

```sh
git clone https://www.github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
conda create -n visdialch python=3.6

# activate the environment and install all dependencies
conda activate visdialch
cd visdial-challenge-starter-pytorch/
pip install -r requirements.txt

# install this codebase as a package in development version
python setup.py develop
```

**Note:** Docker setup is necessary if you wish to extract image features using Detectron.

### Docker

We provide a Dockerfile which creates a light-weight image with all the dependencies installed.

1. Install [nvidia-docker][18], which enables usage of GPUs from inside a container.
2. Build the image as:

```sh
cd docker
docker build -t visdialch .
```

3. Run this image in a container by setting user+group, attaching project root (this codebase) as a volume and setting shared memory size according to your requirements (depends on the memory usage of your model).

```sh
nvidia-docker run -u $(id -u):$(id -g) \
                  -v $PROJECT_ROOT:/workspace \
                  --shm-size 16G visdialch /bin/bash
```

We recommend this development workflow, attaching the codebase as a volume would immediately reflect source code changes inside the container environment. We also recommend containing all the source code for data loading, models and other utilities inside `visdialch` directory. Since it is a setuptools-style package, it makes handling of absolute/relative imports and module resolving less painful. Scripts using `visdialch` can be created anywhere in the filesystem, as far as the current conda environment is active.


Download Data
-------------

1. Download the VisDial v1.0 dialog json files from [here][7] and keep it under `$PROJECT_ROOT/data` directory, for default arguments to work effectively.

2. Get the word counts for VisDial v1.0 train split [here][9]. They are used to build the vocabulary.

3. We also provide pre-extracted image features of VisDial v1.0 images, using a Faster-RCNN pre-trained on Visual Genome. If you wish to extract your own image features, skip this step and download VIsDial v1.0 images from [here][7] instead. Extracted features for v1.0 train, val and test are available for download at these links.

  * [`features_faster_rcnn_x101_train.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_train.h5): Bottom-up features of 36 proposals from images of `train` split.
  * [`features_faster_rcnn_x101_val.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_val.h5): Bottom-up features of 36 proposals from images of `val` split.
  * [`features_faster_rcnn_x101_test.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_test.h5): Bottom-up features of 36 proposals from images of `test` split.

4. We also provide pre-extracted FC7 features from VGG16, although the `v2019` of this codebase does not use them anymore.

  * [`features_vgg16_fc7_train.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_vgg16_fc7_train.h5): VGG16 FC7 features from images of `train` split.
  * [`features_vgg16_fc7_val.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_vgg16_fc7_val.h5): VGG16 FC7 features from images of `val` split.
  * [`features_vgg16_fc7_test.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_vgg16_fc7_test.h5): VGG16 FC7 features from images of `test` split.


Training
--------

This codebase supports both generative and discriminative decoding; read more [here][16]. For reference, we have Late Fusion Encoder from the Visual Dialog paper.

We provide a training script which accepts arguments as config files. The config file should contain arguments which are specific to a particular experiment, such as those defining model architecture, or optimization hyperparameters. Other arguments such as GPU ids, or number of CPU workers should be declared in the script and passed in as argparse-style arguments.

Train the baseline model provided in this repository as:

```sh
python train.py --config-yml configs/lf_disc_faster_rcnn_x101.yml --gpu-ids 0 1 # provide more ids for multi-GPU execution other args...
```

To extend this starter code, add your own encoder/decoder modules into their respective directories and include their names as choices in your config file. We have an `--overfit` flag, which can be useful for rapid debugging. It takes a batch of 5 examples and overfits the model on them.

### Saving model checkpoints

This script will save model checkpoints at every epoch as per path specified by `--save-dirpath`. Refer [visdialch/utils/checkpointing.py][19] for more details on how checkpointing is managed.

### Logging

We use [Tensorboard][5] for logging training progress. Recommended: execute `tensorboard --logdir /path/to/save_dir --port 8008` and visit `localhost:8008` in the browser.


Evaluation
----------

Evaluation of a trained model checkpoint can be done as follows:

```sh
python evaluate.py --config-yml /path/to/config.yml --load-pthpath /path/to/checkpoint.pth --split val --gpu-ids 0
```

This will generate an EvalAI submission file, and report metrics from the [Visual Dialog paper][13] (Mean reciprocal rank, R@{1, 5, 10}, Mean rank), and Normalized Discounted Cumulative Gain (NDCG), introduced in the first Visual Dialog Challenge (in 2018).

The metrics reported here would be the same as those reported through EvalAI by making a submission in `val` phase. To generate a submission file for `test-std` or `test-challenge` phase, replace `--split val` with `--split test`.


Results and pretrained checkpoints
----------------------------------

Performance on `v1.0 test-std` (trained on `v1.0` train + val):

  Model  |  R@1   |  R@5   |  R@10  | MeanR  |  MRR   |  NDCG  |
 ------- | ------ | ------ | ------ | ------ | ------ | ------ |
[lf-disc-faster-rcnn-x101][12] | 0.4617 | 0.7780 | 0.8730 |  4.7545| 0.6041 | 0.5162 |
[lf-gen-faster-rcnn-x101][20]  | 0.3620 | 0.5640 | 0.6340 | 19.4458| 0.4657 | 0.5421 |


Acknowledgements
----------------

* This starter code began as a fork of [batra-mlp-lab/visdial-rl][14]. We thank the developers for doing most of the heavy-lifting.
* The Lua-torch codebase of Visual Dialog, at [batra-mlp-lab/visdial][15], served as an important reference while developing this codebase.
* Some documentation and design strategies of `Metric`, `Reader` and `Vocabulary` classes are inspired from [AllenNLP][17], It is not a dependency because the use-case in this codebase would be too little in its current state.

[1]: https://visualdialog.org/challenge/2019
[2]: https://conda.io/docs/user-guide/install/download.html
[3]: http://images.cocodataset.org/zips/train2014.zip
[4]: http://images.cocodataset.org/zips/val2014.zip
[5]: https://www.github.com/lanpa/tensorboardX
[6]: https://arxiv.org/abs/1707.07998
[7]: https://visualdialog.org/data
[9]: https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/visdial_1.0_word_counts_train.json
[10]: https://visualdialog.org/data
[11]: http://www.robots.ox.ac.uk/~vgg/research/very_deep/
[12]: https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/lf_disc_faster_rcnn_x101_trainval.pth
[13]: https://arxiv.org/abs/1611.08669
[14]: https://www.github.com/batra-mlp-lab/visdial-rl
[15]: https://www.github.com/batra-mlp-lab/visdial
[16]: https://visualdialog.org/challenge/2018#faq
[17]: https://www.github.com/allenai/allennlp
[18]: https://www.github.com/nvidia/nvidia-docker
[19]: https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch/blob/master/visdialch/utils/checkpointing.py
[20]: https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/lf_gen_faster_rcnn_x101_train.pth
