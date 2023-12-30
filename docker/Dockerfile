FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

RUN apt-get update && apt-get install -y python-is-python3 python3-pip python3-opencv

RUN pip3 install --upgrade "jax[cuda12_local]==v0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip3 install flax ml_collections clu && \
    pip3 install tf-models-official