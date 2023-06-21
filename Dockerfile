FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

WORKDIR /workspace

# Basic installs
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ='America/Detroit'
RUN apt-get update -qq \
    && apt-get -y --no-install-recommends install \
       build-essential software-properties-common wget git tar rsync ninja-build \
    && apt-get clean all \
    && rm -r /var/lib/apt/lists/*

# Install Miniconda3 23.3.1
ENV PATH="/root/.local/miniconda3/bin:$PATH"
RUN mkdir -p /root/.local \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py39_23.3.1-0-Linux-x86_64.sh -b -p /root/.local/miniconda3 \
    && rm -f Miniconda3-py39_23.3.1-0-Linux-x86_64.sh \
    && ln -sf /root/.local/miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Install PyTorch
RUN pip install torch==2.0.1

# Install the HEAD commit of Zeus (for ZeusMonitor)
RUN git clone https://github.com/SymbioticLab/Zeus.git zeus \
      && cd zeus \
      && pip install -e . \
      && cd ..

# Install requirements for benchmarking
ADD . /workspace/leaderboard
RUN cd leaderboard \
      && pip install -r requirements.txt \
      && cd ..

ENV TRANSFORMERS_CACHE=/data/leaderboard/hfcache

WORKDIR /workspace/leaderboard
