---
layout: post
title: Parallelism and Optimization in Language Model Training
date: 2022-12-26
description: tricks and tips
tags: Software
categories: Software
---

Modern large language model usually is trained with billions number of parameters and trillions number of tokens. With model size and training data at such scale, computation resource and memory footprint requirement is huge. How to effectively leverage GPU resources to speed up training is an important topic in language model pretraining. In this blog, we'll dive deep into parallel training in recent distributed training paradigms. 

A lot of contents of here are from OpenAI, Nvidia, Deepspeed and bigscience blogs. We'll first go through different parallelism techniques and then talk about how to combine them to maximize training efficiency. 

<p align="center">
    <img alt="gopher dataset" src="/assets/img/speedup.jpg" width="60%"/>
    <br>
</p>


### Data Parallelism
Data parallelism (DP) is the most straightforward way of parallel training. With data parallelism, model parameters and optimzer states are replicated across different workers. Data is partitioned into the same number of shards and each replicate of model is fed with one shard of data. Forward and backward computation is in parallel (simutaneously) and then there is a synchronization step where gradients are averaged across workers to update parameters.

### Pipeline Parallelism
Pipeline parallelism (PP) is from model parallelism. Model parallelism is initially proposed to solve that challenge that one model can't fit into one GPU. The idea is we can vertically slice model into different layers (e.g. one or more layers in transformer models) and put different layers in different GPUs. The issue with this method is that because sequential computation order of layers, if we feed single large batch data into one of the workers, all other workers are idle. This is the so-called `bubble` waiting time.  

To solve the problem, we can reuse the data parallelism idea. Instead of feeding a single large batch into a model shard, we can partition data into small chunks. Each chunk of data goes through different model shards (workers) in a pipeline way. The following figure illustrates how this works. 

<p align="center">
    <img alt="gopher dataset" src="/assets/img/pipeline.png" width="100%"/>
    <br>
    <em>Pipeline parallelism. image from [4]</em>
    <br>
</p>

### Tensor Parallelism
The bottleneck of neural network training is compute. Among all the computation parts, the general matrix multiplication (GEMM) consumes the most of time. One way to parallize the matrix multiplication is to use matrix decomposition. Specifically, we can split a matrix into two or multiple parts based on row or column. Then we can aggregate results after the computation of each parts in the end. This is the core idea of tensor parallelism (TP).


As these three parallelism is orthogonal to each other, it's easy to combine them together. The following diagram shows how to combine pipeline parallelism with data parallelism. 
<p align="center">
    <img alt="dp with pp" src="/assets/img/parallelism-zero-dp-pp.png" width="100%"/>
    <br>
    <em>Combination of pipeline parallelism and data parallelism. Image from Deepspeed tutorial</em>
    <br>
</p>


### ZeRO DP
Zero Redundancy Optimizer (ZeRO) is an optimizied data parallelism proposed by Deepspeed team. The idea is instead of replicating the whole model, optimizer on each of workers, we can only store needed part. 

<p align="center">
    <img alt="zero dp" src="/assets/img/zero.png" width="100%"/>
    <br>
    <em>Zero DP. Image from Deepspeed</em>
    <br>
</p>


## References
[1] https://huggingface.co/blog/bloom-megatron-deepspeed <br>
[2] https://github.com/NVIDIA/NeMo <br>
[3] https://openai.com/blog/techniques-for-training-large-neural-networks/ <br>
[4] [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) <br>
[5] [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) <br>
[6] https://www.deepspeed.ai/tutorials/pipeline/

