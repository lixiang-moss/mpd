# syntax=docker/dockerfile:1.5

ARG CUDA_TAG=11.8.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    sudo \
    locales \
    ca-certificates \
    software-properties-common \
    curl \
    wget \
    git \
    git-lfs \
    build-essential \
    pkg-config \
    cmake \
    ninja-build \
    ffmpeg \
    libegl1 \
    libgl1 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgles2-mesa-dev \
    libglew2.2 \
    libglfw3 \
    libglfw3-dev \
    libglu1-mesa \
    libxcursor-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev \
    libsm6 \
    libvulkan-dev \
    libvulkan1 \
    libx11-6 \
    libx11-dev \
    libxext6 \
    libxkbcommon-x11-0 \
    libxrender1 \
    mesa-common-dev \
    mesa-utils \
    mesa-vulkan-drivers \
    pigz \
    unzip \
    zip \
    vulkan-tools \
    xauth \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN git lfs install --system

# Create non-root user to match VS Code defaults
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# Isaac Gym runtime fixes (borrowed from isaacdocker)
RUN rm -f /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json || true
COPY isaacdocker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY isaacdocker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json

# micromamba for reproducible conda environments
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba \
    && mkdir -p ${MAMBA_ROOT_PREFIX} \
    && chown -R ${USERNAME}:${USERNAME} ${MAMBA_ROOT_PREFIX}
ENV PATH=${MAMBA_ROOT_PREFIX}/bin:${PATH}

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV QT_X11_NO_MITSHM=1

RUN mkdir -p /workspaces/MPDLX-B-new && chown -R ${USERNAME}:${USERNAME} /workspaces
WORKDIR /workspaces/MPDLX-B-new

USER ${USERNAME}

# Keep container alive for VS Code devcontainers
CMD ["sleep", "infinity"]
