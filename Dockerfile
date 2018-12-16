# TODO: update to pytorch 1.0 container when it is avalable
# PyTorch 1.0 will be downloaded through requirements.txt anyway

FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel

COPY . /workspace

RUN pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt

RUN git clone --depth 1 https://www.github.com/facebookresearch/detectron /detectron && \
    pip install -r /detectron/requirements.txt

WORKDIR /detectron
RUN make

WORKDIR /workspace
