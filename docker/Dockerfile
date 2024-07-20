FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

RUN apt-get update && apt-get install -y python-is-python3 python3-pip python3-opencv wget

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install cudnn && \
    apt-get clean && \
    rm cuda-keyring_1.1-1_all.deb

RUN pip3 install --upgrade "jax[cuda12_local]==v0.4.30" && \
    pip3 install flax ml_collections clu && \
    pip3 install tf-models-official
