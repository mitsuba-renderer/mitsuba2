# Start from CUDA 10.2 development image
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Install build tools, including Clang and libc++ (Clang's C++ library)
RUN apt-get update && apt-get install -y \
    clang-9 \
    cmake \
    libc++-9-dev \
    libc++abi-9-dev \
    libjpeg-dev \
    libpng-dev \
    libxcursor-dev \
    libxinerama-dev \
    libxrandr-dev \
    libz-dev \
    ninja-build \ 
    python3-dev \
    python3-distutils \
    python3-setuptools \
    python3-pip

# Install basic Python tools
RUN pip3 install jupyterlab numpy matplotlib ipywidgets

# create /.local so that Python and Jupyter are happy
RUN mkdir /.local && chmod a+rwx /.local

# Set C/C++ compilers to Clang
ENV CC=clang-9
ENV CXX=clang++-9

WORKDIR /mitsuba2
CMD ["/bin/bash"]
