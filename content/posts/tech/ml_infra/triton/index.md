---
title: "Triton, Cuda and GPU"
date: 2024-03-05T00:18:23+08:00
lastmod: 2024-03-05T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- ML
description: "Triton and cuda"
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

Cuda programming is all about multi-threading. Multiple threads create a thread block. Thread block is executed by the stream multiprocessor (SM, the SM-0, SM-1 etc shown below). Threads within the same block has a shared memory which is the shared memory below. Note that L1 cache and shared memory are different. The main difference between shared memory and the L1 is that the contents of shared memory are managed by user code explicitly, whereas the L1 cache is automatically managed. Shared memory is also a better way to exchange data between threads in a block with predictable timing.

Whenever shared memory needs to fetch data from global memory, it first checks whether the data is already in the L2 cache. If it is, then there's no need to access global memory. The higher the probability of a cache hit, the higher the so-called memory hit rate. One of the goals in CUDA programming is to maximize this hit rate as much as possible.

An SM contains multiple subcores, and each subcore has a warp scheduler and dispatcher capable of handling 32 threads.

<div align="center"> <img src=images/gpu_mem_hierarchy.png style="width: 100%; height: auto;"/> image from Nvidia</div>

- Registers—These are private to each thread, which means that registers assigned to a thread are not visible to other threads. The compiler makes decisions about register utilization.
- L1/Shared memory (SMEM)—Every SM has a fast, on-chip scratchpad memory that can be used as L1 cache and shared memory. All threads in a CUDA block can share shared memory, and all CUDA blocks running on a given SM can share the physical memory resource provided by the SM. L1 cache is used by system, and 
- Read-only memory —Each SM has an instruction cache, constant memory,  texture memory and RO cache, which is read-only to kernel code.
- L2 cache—The L2 cache is shared across all SMs, so every thread in every CUDA block can access this memory. The NVIDIA A100 GPU has increased the L2 cache size to 40 MB as compared to 6 MB in V100 GPUs.
- Global memory—This is the framebuffer size of the GPU and DRAM sitting in the GPU


### Common Libs
- Cuda: Library to use GPUs.
- CuTLASS: CUDA GEMM lib.
- CuBLAS: cuda basic linear algebra lib.
- CuDNN: Library to do Neural Net stuff on GPUs (probably uses cuda to talk to the GPUs)



## References


<!-- 4. https://www.zhihu.com/question/613405221/answer/3129776636 -->