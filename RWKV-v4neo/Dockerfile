FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ARG workdir=/workspace
WORKDIR ${workdir}

ENV TZ=Asia/Tokyo
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common tzdata
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y \
    && apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
    liblzma-dev python3-openssl git less
# python
ARG python_version=3.9.13
RUN wget https://www.python.org/ftp/python/${python_version}/Python-${python_version}.tar.xz \
    && tar xJf Python-${python_version}.tar.xz && rm Python-${python_version}.tar.xz
RUN cd Python-${python_version} && ./configure && make && make install && cd ..
RUN wget https://bootstrap.pypa.io/get-pip.py\
    && python3.9 get-pip.py \
    && rm get-pip.py

# python packages
COPY requirements.txt ${workdir}
RUN pip3 install -U pip && pip3 install --no-cache-dir -r requirements.txt

# alias settings
RUN echo 'alias python="python3"' >> ~/.bashrc && \
    echo 'alias pip="pip3"' >> ~/.bashrc && \
    . ~/.bashrc
