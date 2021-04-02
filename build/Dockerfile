FROM nvcr.io/nvidia/pytorch:20.11-py3 
SHELL ["/bin/bash", "-c"]


WORKDIR /workspace/source

ARG HTTP_PROXY
ENV http_proxy ${HTTP_PROXY}
ENV https_proxy ${HTTP_PROXY}

RUN apt update && apt -y upgrade

RUN apt -y install  vim \
  wget \
  curl \
  git \
  gcc \
  cmake \
  sox \
  libsndfile1-dev \
  ffmpeg \
  flac \
  unzip \
  vim \
  tmux 

RUN pip3 install -r requirements.txt

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES utility,compute