# TODO: update to pytorch 1.0 container when it is avalable
# PyTorch 1.0 will be downloaded through requirements.txt anyway

FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

RUN pip install --upgrade pip && pip install cython
RUN git clone --depth 1 https://www.github.com/batra-mlp-lab/visdial-challenge-starter-pytorch /workspace && \
    pip install -r /workspace/requirements.txt

RUN git clone --depth 1 https://www.github.com/facebookresearch/detectron /detectron && \
    pip install -r /detectron/requirements.txt

WORKDIR /detectron
RUN make

WORKDIR /workspace
