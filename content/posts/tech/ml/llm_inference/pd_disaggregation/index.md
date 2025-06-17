---
title: "PD Disaggregation"
date: 2025-04-05T00:18:23+08:00
lastmod: 2025-04-05T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- LLM
description: "Inference"
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
math: true

---

## What is PD Disaggregation

Prefilling decoding disaggregation refers to the process that decouples the prefilling phase and decoding phase in LLM inference. LLM Inference has two stages, prefilling and decoding as we've discussed before. LLM applications often emphasize individual latency for each phase: time to first token (TTFT) for the prefilling phase and time per output token (TPOT) of each request for the decoding phase. To provide good customer experience, different application optimizes different target. For instance, real-time chatbot system prioritizes low TTFT while DeepResearch type of application wants to reduce the TPOT such that the whole generation time would be shorter. 

Decoupling the prefilling and decoding offers the flexibility to optimize each stage in decoding, thus has shown improving GPU utilization.

## KV Cache Store
Now we understand that why we need to separate the two stages of prefilling and decoding. In practice, we'll have to maintain two full set of models on different GPUs. After we compute the prefix, we transfer it to the decoding stage. The two stages can be seen as the producer and consumer. Like microservice system where we decouple client requests with backend services using message queue, here we can use KV cache store as the intermediate layer. 

<div align="center"> <img src=images/mooncake.png style="width: 100%; height: auto;"/> </div>

In Mooncake [2] inference architecture, there is kv cache store which is the kv cache pool shown above. The nice thing about adding kv cache store is that if decoding stage crashes, the system won't need to go through the prefilling stage again thus has better fault resilience. 

The kv cache store can also be used for prefix caching. 



## Computation and Communication Overlapping

One thing that distinguishes KV cache from other type of caches is its huge size, thus it's hard to store it using distributed caching service such as Redis/Memcache. To put things into perspective, we can do a simple math here: assuming we have a llama 70B model which has 80 layers. 
Assuming using BF16, then the total kv cache size of 1024 prefix is: 

512(BS) * 1024( L:prefix ) * 8(D) * 128(H) * 80 (layer) * 2 (BF16) * 2 (KV) = 160GB !!!

Directly transferring such large amount of data is slow. We can overlap the communication with computation to save time. After computation of KV cache for each layer, we can start transfer the cache to the KV cache store. Mooncake's utilize this implementation. 



### References
1. [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670)
2. [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079)
    https://github.com/kvcache-ai/Mooncake
3. DeepSeek-V3 Technical Report
4. P/D-Serve: Serving Disaggregated Large Language Model at Scale


<!-- 4. https://zhuanlan.zhihu.com/p/27836625742 -->
<!-- https://github.com/chen-ace/LLM-Prefill-Decode-Benchmark/blob/main/%E5%AE%9E%E9%AA%8C1-%E6%8F%AD%E7%A4%BAPD%E5%88%86%E7%A6%BB%E5%8E%9F%E5%9B%A0/experiment_pd_cuda.py -->
<!-- https://yangwenbo.com/articles/llm-prefix-caching.html -->
<!-- https://zhuanlan.zhihu.com/p/23081000392 -->