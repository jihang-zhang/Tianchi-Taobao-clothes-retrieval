FROM nvidia/cuda:10.1-cudnn7-devel

ADD . /

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	libsm6 libxext6 libxrender-dev ca-certificates \
	python3-dev build-essential pkg-config git curl wget automake libtool sudo \
	cmake protobuf-compiler libprotobuf-dev && \
  rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# install dependencies
RUN pip install numpy pandas tqdm PyYAML opencv-python

# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install tensorboard cython
RUN pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install 'git+https://github.com/facebookresearch/fvcore'
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN pip install timm==0.1.18

# install detectron2
RUN git clone https://github.com/jihang-zhang/detectron2 detectron2_repo
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
RUN pip install -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /

# Execute
CMD ["sh", "run.sh"]
