FROM nvcr.io/nvidia/jax:25.01-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

RUN apt-get update && apt-get install -y python-is-python3 python3-pip python3-opencv wget

RUN pip3 install clu && \
    pip3 install tf-models-official
