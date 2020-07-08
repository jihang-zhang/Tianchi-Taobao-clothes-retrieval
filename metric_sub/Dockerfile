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
RUN pip install numpy pandas tqdm PyYAML opencv-python albumentations scikit-image feather-format transformers
RUN pip install -U scikit-learn

# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install tensorboard cython
RUN pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html

RUN pip install 'git+https://github.com/facebookresearch/fvcore'
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN pip install timm

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /

# Execute
CMD ["sh", "run.sh"]
