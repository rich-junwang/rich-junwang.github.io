---
title: "CUDA"
date: 2023-05-05T00:18:23+08:00
lastmod: 2023-05-05T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- distributed training
description: "Distributed data processing"
weight:
slug: ""
draft: false # 是否为草稿
comments: true
reward: false # 打赏
mermaid: true #是否开启mermaid
showToc: true # 显示目录
TocOpen: true # 自动展开目录
hidemeta: false # 是否隐藏文章的元信息，如发布日期、作者等
disableShare: true # 底部不显示分享栏
showbreadcrumbs: true #顶部显示路径
cover:
    image: "" #图片路径例如：posts/tech/123/123.png
    caption: "" #图片底部描述
    alt: ""
    relative: false
---

## CUDA Ecosystem

### CUDA Toolkit (Libraries and Compiler)
The CUDA Toolkit contains the NVCC compiler and the development libraries (like cuBLAS, cuDNN, and the core CUDA libraries). Those libraries build on top of runtime API to save developers from writing complex, performance-critical algorithms from scratch.

CUDA Toolkit Installation

Go to the link [here](https://developer.nvidia.com/cuda-toolkit). It will guide you to the following instructions. Note that select the cuda-toolkit version lower than the cuda driver version (from nvidia-smi).
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0
```

After the installation, adding the binary path into your PATH
```bash
export PATH=${PATH}:/usr/local/cuda/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
```


Check installation,
```bash
nvcc --version
```

If we have multiple cuda installed, we can use the following command to choose the default one
```bash
sudo update-alternatives --display cuda

# If CUDA 12.8 is not listed as an option, add it:
sudo update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-12.8 128

# now select it
sudo update-alternatives --config cuda


# we can also manually create the symlink
sudo rm /etc/alternatives/cuda
sudo ln -s /usr/local/cuda-12.8 /etc/alternatives/cuda
sudo rm /usr/local/cuda
sudo ln -s /etc/alternatives/cuda /usr/local/cuda

# confirm
ls -al /usr/local/cuda
ls -al /etc/alternatives/cuda

# should display
/usr/local/cuda -> /etc/alternatives/cuda
/etc/alternatives/cuda -> /usr/local/cuda-12.8
```

### CUDA Runtime API

The CUDA Runtime API is a set of functions that allow applications to interact with the GPU. The PyTorch binary we install already includes its own version of the necessary CUDA runtime files. Another example is `cuda_runtime.h` which we commonly used in cuda programming. 

The diagram below shows the main components in cuda development toolkit. 

When we install packages like flash-attn, usually what we talk about is the CUDA version is CUDA toolkit version.

<div align="center"> <img src=images/cuda.svg style="width: 60%; height: auto;"/> </div>


`nvidia-smi` shows the nvidia driver’s runtime version, NOT the CUDA libraries needed to compile or run GPU frameworks


In terms of compatibility, a newer NVIDIA driver (e.g., supporting CUDA 12.4) is generally backward-compatible with older CUDA Toolkits (e.g., CUDA 11.8).




## Two-Stage Compilation of NVCC

When we run CUDA codes, it goes through Source Code (CUDA C++) → PTX → CUBIN → GPU Execution. Essentially NVCC adopts a two-stage compilation process. 

1. PTX (Parallel Thread Execution) — a virtual GPU architecture
2. CUBIN (SASS) — actual hardware-specific binary for real GPUs

### Stage 1 — NVCC compiles CUDA code into PTX

```
nvcc -arch=compute_80 kernel.cu → kernel.ptx
```

Here we choose a *virtual architecture* such as:

* `compute_50`
* `compute_70`
* `compute_90`

These are *not real GPU chips*. They describe what CUDA features your code is allowed to use.

### Stage 2 — PTX to machine code (CUBIN)

```
nvcc -code=sm_80 kernel.cu → kernel.cubin
```

OR (if PTX is bundled):

```
NVIDIA driver JIT → sm_86 or future GPU
```

Real GPUs have machine code defined by their SM architecture:

* `sm_50`, `sm_70`, `sm_80`, `sm_90`, etc.

These architectures change with each GPU generation.


### PTX

PTX (Parallel Thread Execution) is an intermediate representation (IR)

* A virtual GPU instruction set
* Abstract, stable, independent of real hardware
* Easy for compilers to target
* Similar to LLVM IR

PTX is the secret that makes CUDA programs run on future GPUs. If CUDA only emitted native machine code, your program compiled in 2016 for Pascal (sm_60) would not run on a 2023 GPU like Ada Lovelace (sm_89).

But thanks to PTX:

* Old binaries contain PTX
* Driver JIT compiles PTX → new architecture
* Your program keeps working without recompilation

So, we can say that PTX = Forward Compatibility. CUDA chose a two-stage model for maintainability, performance, and long-term compatibility.






## References
1. https://vutr.substack.com/p/the-overview-of-parquet-file-format
<!-- https://zhuanlan.zhihu.com/p/675767714 -->
