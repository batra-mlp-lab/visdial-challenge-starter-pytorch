# TODO: update to pytorch 1.0 container when it is avalable
# PyTorch 1.0 will be downloaded through requirements.txt anyway

FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel

COPY . /workspace

RUN pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt

EXPOSE 8008
