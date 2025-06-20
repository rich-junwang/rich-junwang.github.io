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

<div align="center"> <img src=images/gpu_mem_hierarchy.png style="width: 100%; height: auto;"/> Figure 1. image from Nvidia</div>

- Registers—These are private to each thread, which means that registers assigned to a thread are not visible to other threads. The compiler makes decisions about register utilization.
- L1/Shared memory (SMEM)—Every SM has a fast, on-chip scratchpad memory that can be used as L1 cache and shared memory. All threads in a CUDA block can share shared memory, and all CUDA blocks running on a given SM can share the physical memory resource provided by the SM. L1 cache is used by system, and 
- Read-only memory —Each SM has an instruction cache, constant memory,  texture memory and RO cache, which is read-only to kernel code.
- L2 cache—The L2 cache is shared across all SMs, so every thread in every CUDA block can access this memory. The NVIDIA A100 GPU has increased the L2 cache size to 40 MB as compared to 6 MB in V100 GPUs.
- Global memory—This is the framebuffer size of the GPU and DRAM sitting in the GPU


## Cuda Programming
When we talk about cuda GPU parallel computing, we are actually referring to a heterogeneous computing architecture based on both CPU and GPU. In this architecture, the GPU and CPU are connected via a PCIe bus to work together collaboratively. The CPU side is referred to as the **host**, while the GPU side is referred to as the **device**.


The CUDA programming requires cooperation between the CPU and GPU. In CUDA, the term host refers to the CPU and its memory, while device refers to the GPU and its memory. A CUDA program includes both host code and device code, which run on the CPU and GPU respectively. Additionally, the host and device can communicate with each other, allowing data to be transferred between them.

The typical execution flow of a CUDA program is as follows:

- Allocate host memory and initialize the data;
- Allocate device memory and copy data from the host to the device;
- Launch a CUDA kernel to perform computations on the device;
- Copy the results from the device back to the host;
- Free the memory allocated on both the device and the host.


A kernel is a function that runs in parallel across multiple threads on the device (GPU). Kernel functions are declared using the __global__ qualifier, and when calling a kernel, the syntax \<\<\<grid, block\>\>\> is used to specify the number of threads to execute. The nvcc compiler recognizes this modifier and splits the code into two parts, sending them to the CPU and GPU compilers respectively for compilation. 

The \<\<\<a, b\>\>\> syntax following the kernel function call is a special CUDA syntax. These two numbers represent the number of thread blocks and the number of threads per block used during the kernel function execution. Since each thread block and each thread operate in parallel, this allocation determines the degree of parallelism in the program. In our case, since there is only one computation, we only assigned one thread within a single block. 

All the threads launched by a kernel are collectively called a grid. Threads within the same grid share the same global memory space.A grid can be divided into multiple thread blocks (blocks), and each block contains many threads.

Below both the grid and the block are 2-dimensional. Both grid and block are defined as variables of type `dim3`. The `dim3` type can be thought of as a struct containing three unsigned integer members: `x`, `y`, and `z`, which are initialized to 1 by default. Therefore, grid and block can be flexibly defined as 1-dimensional, 2-dimensional, or 3-dimensional structures. When calling the kernel, the execution configuration `<<<grid, block>>>` must be used to specify the number and structure of threads that the kernel will use.

A thread requires two built-in coordinate variables (`blockIdx` and `threadIdx`) to be uniquely identified. Both are variables of type `dim3`. 


<div align="center"> <img src=images/kernel.png style="width: 80%; height: auto;"/> Figure 2. kernel 2-dim structure</div>



In CUDA, every thread executes a kernel function, and each thread is assigned a unique thread ID. This thread ID can be accessed within the kernel using the built-in variable threadIdx.

```c
#include<stdio.h>

__global__ void kernel(int a, int b, int *c){
	*c = a + b;
}

int main(){
	int c = 20;
	int *c_cuda;
	cudaMalloc((void**)&c_cuda,sizeof(int));
	kernel<<<1,1>>>(1,1,c_cuda);
	cudaMemcpy(&c,c_cuda,sizeof(int),cudaMemcpyDeviceToHost);
	printf("c=%d\n",c);
	cudaFree(c_cuda);
	return 0;
}
```
The CPU can pass `c_cuda` as a parameter and perform type conversions, but it absolutely cannot read from or write to `c_cuda`, because this variable was allocated using `cudaMalloc` and therefore resides in GPU memory, not system memory. Similarly, the GPU cannot access the variable `c`. The bridge between the two is the `cudaMemcpy` function, which transfers values back and forth over the data bus. This essentially forms a logical structure where the CPU is responsible for sending and receiving data, while the GPU handles the computation.




### Common Libs
- Cuda: Library to use GPUs.
- CuTLASS: CUDA GEMM lib.
- CuBLAS: cuda basic linear algebra lib.
- CuDNN: Library to do Neural Net stuff on GPUs (probably uses cuda to talk to the GPUs)

<div align="center"> <img src=images/gemm_cuda.png style="width: 100%; height: auto;"/> image from [1]</div>



## References

1. https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
2. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model
3. https://zhuanlan.zhihu.com/p/34587739
<!-- https://www.zhihu.com/question/613405221/answer/3129776636 -->

<!-- https://zhuanlan.zhihu.com/p/482238286 -->