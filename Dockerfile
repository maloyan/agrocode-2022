#FROM python:3.9-buster
#FROM nvcr.io/nvidia/cuda:11.4.2-devel-ubuntu20.04 as base
FROM nvcr.io/nvidia/pytorch:22.08-py3

LABEL PROJECT="agrocode"

ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Python minor version is not set, since we use ubuntu 20.04 with default python3.8

RUN apt-get update && apt-get install -y --no-install-recommends \
  software-properties-common \
  curl \
  unzip \
  python3 \
  python3-pip \
  libgl1-mesa-glx \
  wget \
  git \
    && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /builds
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs


COPY ./ /app
WORKDIR /app
# RUN git clone https://huggingface.co/openai/clip-vit-base-patch32 && cd clip-vit-base-patch32 && git lfs install && git pull
RUN echo "import timm; timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=True, num_classes=0); timm.create_model('convnext_xlarge_384_in22ft1k', pretrained=True, num_classes=0); timm.create_model('convnext_xlarge_in22ft1k', pretrained=True, num_classes=0); timm.create_model('convnext_xlarge_in22k', pretrained=True, num_classes=0)" > tmp.py && python tmp.py
RUN echo "import os; from clip.clip import _MODELS, _download; model_path = _download(_MODELS['ViT-L/14@336px'], os.path.expanduser('~/.cache/clip'))" > tmp.py && python tmp.py

CMD [ "python", "-u", "baseline.py" ]
