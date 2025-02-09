# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


#ARG BASE_IMAGE=nvcr.io/nvidia/l4t-base:r35.2.1
#FROM ${BASE_IMAGE}
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3


ENV DEBIAN_FRONTEND=noninteractive



#
# install prerequisites
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            python3-pip \
		  python3-dev \
		  libopenblas-dev \
		  libopenmpi-dev \
            openmpi-bin \
            openmpi-common \
		  gfortran \
		  libomp-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip3 install --no-cache-dir setuptools Cython wheel
RUN pip3 install --no-cache-dir --verbose numpy
RUN pip3 install --no-cache-dir --verbose onnx


#
# PyTorch (these args get set by the build script)
#
ARG PYTORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"
ARG PYTORCH_WHL="torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"

RUN wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${PYTORCH_URL} -O ${PYTORCH_WHL} && \
    pip3 install --no-cache-dir --verbose ${PYTORCH_WHL} && \
    rm ${PYTORCH_WHL}

RUN python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(torch.__config__.show());'


#
# torchvision
#
ARG TORCHVISION_VERSION="v0.15.1"
ARG TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2;8.7"

RUN printenv && echo "torchvision version = $TORCHVISION_VERSION" && echo "TORCH_CUDA_ARCH_LIST = $TORCH_CUDA_ARCH_LIST"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  git \
		  build-essential \
		  ninja-build \
            libjpeg-dev \
		  zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN git clone --branch ${TORCHVISION_VERSION} --recursive --depth=1 https://github.com/pytorch/vision torchvision && \
    cd torchvision && \
    git checkout ${TORCHVISION_VERSION} && \
    python3 setup.py install && \
    cd ../ 
    #rm -rf torchvision

# note:  this was used on older torchvision versions (~0.4) to workaround a bug,
#        but has since started causing another bug as of torchvision 0.11.1
# ARG PILLOW_VERSION=pillow<7    
# pip3 install --no-cache-dir "${PILLOW_VERSION}"


# 
# upgrade cmake - https://stackoverflow.com/a/56690743
# this is needed for newer versions of torchaudio (>= v0.10)
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  software-properties-common \
		  apt-transport-https \
		  ca-certificates \
		  gnupg \
		  lsb-release \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
		  
# typically --only-upgrade is used in the apt install, but cmake wasn't previously installed in this container	
# note: skipping this way for now, because having trouble pinning it to specific version (see below)
# also avoid kitware's rotating GPG keys: https://github.com/dusty-nv/jetson-containers/issues/216
#RUN wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add - && \
#    apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
#    apt-get update && \
#    apt-cache policy cmake && \
#    apt-get install -y --no-install-recommends \
#            cmake \
#    && rm -rf /var/lib/apt/lists/* \
#    && apt-get clean
    
# note:  cmake is currently pinned to 3.22.3 because of https://github.com/pytorch/pytorch/issues/74955	 
RUN pip3 install --upgrade --no-cache-dir --verbose cmake
RUN cmake --version


# patch for https://github.com/pytorch/pytorch/issues/45323
RUN PYTHON_ROOT=`pip3 show torch | grep Location: | cut -d' ' -f2` && \
    TORCH_CMAKE_CONFIG=$PYTHON_ROOT/torch/share/cmake/Torch/TorchConfig.cmake && \
    echo "patching _GLIBCXX_USE_CXX11_ABI in ${TORCH_CMAKE_CONFIG}" && \
    sed -i 's/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=")/  set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")/g' ${TORCH_CMAKE_CONFIG}

    
#
# torchaudio
#
ARG TORCHAUDIO_VERSION="v0.13.1"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  pkg-config \
		  libffi-dev \
		  libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip3 install --no-cache-dir scikit-build && \
    pip3 install --no-cache-dir ninja && \
    pip3 install --no-cache-dir --verbose pysoundfile

# note:  see https://github.com/pytorch/audio/issues/2295 for the reason for the sed commands
RUN git clone --branch ${TORCHAUDIO_VERSION} --recursive --depth=1 https://github.com/pytorch/audio torchaudio && \
    cd torchaudio && \
    git checkout ${TORCHAUDIO_VERSION} && \
    sed -i 's#  URL https://zlib.net/zlib-1.2.11.tar.gz#  URL https://github.com/madler/zlib/archive/refs/tags/v1.2.12.tar.gz#g' third_party/zlib/CMakeLists.txt || echo "failed to patch torchaudio/third_party/zlib/CMakeLists.txt" && \
    sed -i 's#  URL_HASH SHA256=c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1#  URL_HASH SHA256=d8688496ea40fb61787500e863cc63c9afcbc524468cedeb478068924eb54932#g' third_party/zlib/CMakeLists.txt || echo "failed to patch torchaudio/third_party/zlib/CMakeLists.txt" && \
    BUILD_SOX=1 python3 setup.py install && \
    cd ../
    #rm -rf torchaudio


# 
# install OpenCV (with CUDA)
#

#ARG OPENCV_URL=https://nvidia.box.com/shared/static/5v89u6g5rb62fpz4lh0rz531ajo2t5ef.gz
#ARG OPENCV_DEB=OpenCV-4.5.0-aarch64.tar.gz

#COPY scripts/opencv_install.sh /tmp/opencv_install.sh
#RUN cd /tmp && ./opencv_install.sh ${OPENCV_URL} ${OPENCV_DEB}

# 1) ビルドに必要な依存パッケージをインストール
RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    libgtk2.0-dev pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev libavutil-dev libswscale-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    python3-dev python3-pip \
    # もしOpenCVでJPEG/TIFF/PNG等を扱うなら対応ライブラリも:
    libjpeg-dev libpng-dev libtiff-dev \
    # For optional modules
    libdc1394-22-dev libv4l-dev \
    && rm -rf /var/lib/apt/lists/*

 # ソースダウンロード
RUN git clone --branch 4.5.0 https://github.com/opencv/opencv.git /tmp/opencv

RUN git clone --branch 4.5.0 https://github.com/opencv/opencv_contrib /tmp/opencv_contrib

# 3) OpenCVをビルド (CUDA有効, cudevモジュールを含む)
RUN mkdir -p /tmp/opencv/build && cd /tmp/opencv/build && \
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_CUDA=ON \
    -DCUDA_ARCH_BIN="7.2" \
    -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
    -DBUILD_opencv_cudacodec=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_python3=ON \
    -DPYTHON3_EXECUTABLE=/usr/bin/python3 \
    -DPYTHON3_INCLUDE_DIR=/usr/include/python3.8 \
    -DPYTHON3_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.8.so \
    -DPYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
    -DPYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 \
    && make -j$(nproc) && make install && ldconfig

RUN rm -rf /tmp/opencv /tmp/opencv_contrib


#
# PyCUDA
#
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN echo "$PATH" && echo "$LD_LIBRARY_PATH"

RUN pip3 install --no-cache-dir --verbose pycuda six


#
# if needed, patch PyTorch version string to be compliant with PEP 440
#
#RUN if [ -d "/usr/local/lib/python3.8/dist-packages/torch-2.0.0.nv23.05.dist-info" ]; then \
#     echo "patching PyTorch version string to be PEP 440 compliant..."; \
#	sed -i 's/2.0.0.nv23.05/2.0.0/g' /usr/local/lib/python3.8/dist-packages/torch/version.py; \
#	sed -i 's/2.0.0.nv23.05/2.0.0/g' /usr/local/lib/python3.8/dist-packages/torch-2.0.0.nv23.05.dist-info/METADATA; \
#	head /usr/local/lib/python3.8/dist-packages/torch/version.py; \
#	head /usr/local/lib/python3.8/dist-packages/torch-2.0.0.nv23.05.dist-info/METADATA; \
#    fi

# h5py ビルドに必要なパッケージをインストール

RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    g++ \
   && rm -rf /var/lib/apt/lists/*

# libgomp をインストールする（例）
#RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# LD_PRELOAD を常に適用
#ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp-d22c30c5.so.1.0.0
#ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 

#ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

#FROM python:3.8
# 1) Pythonパッケージの更新
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

RUN pip uninstall opencv-python opencv-contrib-python opencv-python-headless

# 2) 顔検出・ランドマークなど
#    - retina-face (PyTorch実装) は複数フォークがあるため注意
#    - face-alignment: 2D/3Dランドマーク (FANベース)
#    - tqdm, opencv-python など、必要に応じて追加
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-scipy \
        #python3-opencv \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip3 install --no-cache-dir \
    matplotlib \
    face-alignment \
    #retina-face \
    tqdm

WORKDIR /app
RUN git clone https://github.com/biubug6/Pytorch_Retinaface.git /app/Pytorch_Retinaface




# pip upgrade & install 
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir mediapipe 
RUN pip3 install --no-cache-dir scikit-image scikit-learn flask


RUN pip3 uninstall -y opencv-python opencv-contrib-python opencv-python-headless



#    あるいは学術モデルをクローン:
# 2) JAANet リポジトリをクローン
#    /opt/jaanet にソースコードを配置 (任意のディレクトリ可)
RUN git clone https://github.com/ZhiwenShao/PyTorch-JAANet.git /opt/jaanet
# (以降、学術モデルのビルド・学習済みモデルの配置など)

RUN python3 -c "import cv2; print(cv2.__file__)"

#
# ワークディレクトリ & ソースコード配置
#
# copy JAANet code & weights
# 3) 作業ディレクトリ
WORKDIR /app

# 4) ソースコード & 重みファイルをコピー
#    - "jaanet_weights" 全体
COPY jaanet_weights /app/jaanet_weights
COPY templates/index.html /app/templates/index.html

#    - "run_jaanet_realtime.py" + その他Pythonスクリプト
COPY run_jaanet_realtime.py /app/
COPY check_mat.py /app/
COPY face_alignment_test.py /app/
COPY landmark_blink_gaze_au.py /app/
COPY main_app.py /app/
COPY main_app2.py /app/
COPY my_script.py /app/
COPY network.py /app/
COPY best_model.pth /app/
COPY best_model2.pth /app/


# (5) Pytorch_Retinaface フォルダをコピー
COPY Pytorch_Retinaface /app/Pytorch_Retinaface

#    - reflect_66.mat
COPY reflect_66.mat /app/
COPY app_test.py /app/

# 5) 動画ファイル or テスト用ファイルをコピー (任意)
#COPY IMG_7721.mp4 /app/
#COPY WIN_20250115_17_01_27_Pro2.mp4 /app/
COPY output /app/output

ENV PYTHONPATH="/app/Pytorch_Retinaface:${PYTHONPATH}"

# 6) (任意) デフォルトコマンド
#    ここでは一旦bashにしておき、docker run時に実行コマンドを指定
CMD ["/bin/bash"]