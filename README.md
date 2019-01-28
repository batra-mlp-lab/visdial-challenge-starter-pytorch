Visual Dialog Challenge Starter Code
====================================

PyTorch starter code for the [Visual Dialog Challenge 2019][1].

  * [Setup and Dependencies](#setup-and-dependencies)
  * [Download Preprocessed Data](#download-preprocessed-data)
  * [Training](#training)
  * [Evaluation](#evaluation)
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


Setup and Dependencies
----------------------

This starter code is implemented using PyTorch v1.0 and provides out of the box support with CUDA 9 and CuDNN 7. There are two recommended ways to set up this codebase: Anaconda or Miniconda, and Docker.

### Anaconda or Miniconda

1. Install Anaconda or Miniconda distribution based on Python3+ from their [downloads' site][2].
2. Clone this repository and create an environment:

```sh
git clone https://www.github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
conda create -n visdialch python=3.6

# activate the environment and install all dependencies
conda activate visdialch
pip install -r requirements.txt

# install this codebase as a package in development version
python setup.py develop
```

**Note:** Docker setup is necessary if you wish to extract image features using Detectron.

### Docker

1. Install [nvidia-docker][18], which enables usage of GPUs from inside a container.
2. We provide a Dockerfile which creates a light-weight image with all the dependencies installed. Build the image as:

```sh
cd docker
docker build -t visdialch .
```

3. Run this image in a container by setting current user, attaching project root (this codebase) as a volume and setting shared memory size according to your requirements (depends on the memory usage of your code).

```sh
nvidia-docker run -u $(id -u):(id -g) -v $PROJECT_ROOT:/workspace \
           --shm-size 16G visdialch /bin/bash
```

Since the codebase is attached as a volume, any changes made to the source code from outside the container will be reflected immediately inside the container, hence this would fit easily in almost any development workflow. We recommend you to contain all the source code for data loading, models and other utilities inside `visdialch` directory, since it is a setuptools-style package, it makes handling of absolute/relative imports and module resolving less painful. Scripts using `visdialch` can be created anywhere in the filesystem, as far as the current conda environment is active.


Download Preprocessed Data
--------------------------

We provide pre-extracted image features of VisDial v1.0 images, extracted from VGG16, and Faster-RCNN. If you wish to extract your own image features, skip this step.

Extracted features for v1.0 train, val and test are available for download at these links. The detector is a Faster-RCNN with ResNeXt-101 backbone, pre-trained on Visual Genome images.

* [`features_faster_rcnn_x101_train.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_train.h5): Bottom-up features of 36 proposals from images of `train` split.
* [`features_faster_rcnn_x101_val.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_val.h5): Bottom-up features of 36 proposals from images of `val` split.
* [`features_faster_rcnn_x101_test.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_test.h5): Bottom-up features of 36 proposals from images of `test` split.
* [`features_vgg16_fc7_train.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_vgg16_fc7_train.h5): VGG16 FC7 features from images of `train` split.
* [`features_vgg16_fc7_val.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_vgg16_fc7_val.h5): VGG16 FC7 features from images of `val` split.
* [`features_vgg16_fc7_test.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_vgg16_fc7_test.h5): VGG16 FC7 features from images of `test` split.

Download these files to `data` directory.


Training
--------

This codebase supports discriminative decoding only; read more [here][16]. For reference, we have Late Fusion Encoder from the Visual Dialog paper.

We provide a training script which accept arguments as config files. The config file should contain arguments which are specific to a particular experiment, such as those defining model architecture, or optimization hyperparameters. Other arguments such as GPU ids, or number of CPU workers should be declared in the script and passed in as argparse-style arguments.

Train the baseline model provided in this repository as:

```sh
python train.py --config-yml configs/lf_disc_vgg16_fc7_bs32.yml --gpu-ids 0 1 # provide more ids for multi-GPU execution other args...
```

To extend this starter code, add your own encoder/decoder modules into their respective directories and include their names as choices in your config file. We have an `--overfit` flag, which can be useful for rapid debugging. It takes a batch of 5 examples and overfits the model on them.

**Saving model checkpoints:** This script will save model checkpoints at every epoch as per path specified by `--save-dirpath`. We recommend you to read the module docstring in [visdialch/utils/checkpointing.py][19] for more details on how checkpointing is managed.


Evaluation
----------

Evaluation of a trained model checkpoint can be done as follows:

```sh
python evaluate.py --config-yml /path/to/config.yml --load-pthpath /path/to/checkpoint.pth --split val --gpu-ids 0
```

This will generate an EvalAI submission file, and report metrics from the [Visual Dialog paper][13] (Mean reciprocal rank, R@{1, 5, 10}, Mean rank), and Normalized Discounted Cumulative Gain (NDCG), introduced in the first Visual Dialog Challenge (in 2018).

**Note:** The metrics reported here would be the same as those reported through EvalAI by making a submission in `val` phase.

To generate a submission file for `test-std` or `test-challenge` phase, replace `--split val` with `--split test`.


Acknowledgements
----------------

* This starter code began as a fork of [batra-mlp-lab/visdial-rl][14]. We thank the developers for doing most of the heavy-lifting.
* The Lua-torch codebase of Visual Dialog, at [batra-mlp-lab/visdial][15], served as an important reference while developing this codebase.
* Some documentation and design strategies of `Metric`, `Reader` and `Vocabulary` classes are inspired from [AllenNLP][17], It is not a dependency because the use-case in this codebase would be too little in its current state.

[1]: https://visualdialog.org/challenge/2019
[2]: https://conda.io/docs/user-guide/install/download.html
[3]: http://images.cocodataset.org/zips/train2014.zip
[4]: http://images.cocodataset.org/zips/val2014.zip
[10]: https://visualdialog.org/data
[11]: http://www.robots.ox.ac.uk/~vgg/research/very_deep/
[13]: https://arxiv.org/abs/1611.08669
[14]: https://www.github.com/batra-mlp-lab/visdial-rl
[15]: https://www.github.com/batra-mlp-lab/visdial
[16]: https://visualdialog.org/challenge/2018#faq
[17]: https://www.github.com/allenai/allennlp
[18]: https://www.github.com/nvidia/nvidia-docker
[19]: https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch/blob/master/visdialch/utils/checkpointing.py
