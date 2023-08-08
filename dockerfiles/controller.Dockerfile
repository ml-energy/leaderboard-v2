FROM ubuntu:22.04

# Basic installs
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ='America/Detroit'
RUN apt-get update -qq \
    && apt-get -y --no-install-recommends install \
       software-properties-common wget git \
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

# Install spitfight
ADD . /workspace/leaderboard
RUN cd /workspace/leaderboard \
      && pip install -e .[colosseum-controller]

WORKDIR /workspace/leaderboard

CMD ["uvicorn", "spitfight.colosseum.controller.router:app", "--host", "0.0.0.0"]
