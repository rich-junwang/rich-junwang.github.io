---
title: "vLLM"
date: 2024-10-05T12:01:14-07:00
lastmod: 2024-10-05T12:01:14-07:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- Inference
description: "VLLM"
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

## LLM Inference Modes
vLLM has two inference modes:
- offline batch mode: mostly for offline model evaluation, large-scale, high-throughput inference where latency is less critical 
- online serving mode: Real-time applications like chatbots or APIs where latency is important.


## LLM Inference
No matter what kind mode vLLM is in for inference, the LLM inference process has two stages: 
- prefilling
- decoding

Prefilling phase llm encodes the prompt at once. This is one forward path execution in LLM. 
Decoding phase, llm generates each token step by step. 

## KV Cache

KV Cache at inference time refers that caching key and value vectors in self-attention saves redundant computation and accelerates decoding - but takes up memory.



## Page Attention

Traditional LLM inference has inefficient KV cache management. There are three issues:
 - reserved slots for future tokens: although the slots are used eventually, the pre-allocation leads to most of time the memory slots are wasted . Think about the end token slot is empty in the whole decoding process.  
 - Internal fragmentation due to over-provisioning for potential maximum sequence lengths,
 - External fragmentation from the memory allocator like the buddy allocator.


Remember that in OS, each program sees isolated, contiguous logical address (virtual memory) ranging from 0 - N which is dynamically mapped to physical memory (main memory). 

<div align="center"> <img src=images/mem.png style="width: 80%; height: auto;"/> </div>

Because of the indirection between logical and physical addresses, a process’s pages can be located anywhere in memory and can be moved (relocated). Needed pages are ‘paged into’ main memory as they are referenced by the executing program. 

Designed with a similar philosophy, paged attention utilizes logical view and physical view to manage KV cache to minimize fragmentation in GPU memory.

<div align="center"> <img src=images/pagedattn.png style="width: 80%; height: auto;"/> </div>

Obviously the block table here is the same with memory management unit which manages the mapping from logical address to physical address. For page attention here logical view is just needed batch KV cache. 


## Continuous Batching
The main topic for inference is how to handle multiple concurrent requests efficiently. Remember that LLM inference is memory-IO bound, not compute-bound. This means it takes more time to load 1MB of data to the GPU’s compute cores than it does for those compute cores to perform LLM computations on 1MB of data. Thus LLM inference throughput is largely determined by how large a batch you can fit into high-bandwidth GPU memory.

The very obvious one for online serving is to batch processing individual request. The left side of the following figure shows request-level batching. However as is illustrated in the figure, GPU is underutilized as generation lengths vary. 

As is shown in the right hand side, in continuous batching, LLMs continuously add new requests to the current batch as others finish. This allows it to maximize GPU utilization. In reality, the complexity comes from how to manage the KV cache of requests which are finished. In vLLM, KV cache is managed by pagedattention discussed above which makes things slightly easier. 

<div align="center"> <img src=images/batching.png style="width: 80%; height: auto;"/> </div>


## AsyncLLM

When a request comes in, it goes through a sequence of steps: 

1. tokenization: tokenizing the prompt, 
2. prefilling: loading it into the model cache (prefill), 
3. decoding: generating tokens one by one (decode).


In a naïve synchronous setup, it would process each request’s entire prefill → decode phases before starting the next, leaving the GPU idle whenever that one request is waiting (e.g., on small decode batches) or blocked on I/O.

By contrast, an asynchronous engine interleaves these phases across many requests. While Request A is in its decode phase waiting for the next iteration, Request B’s prefill can run, or Request C’s decode can proceed. This “overlap” of lifecycles means the GPU is never sitting idle between operations, because there’s always some request at the right stage to feed it compute work

To utilize the AsyncLLMEngine, we can instantiate it as follows. This simple instantiation allows you to start making asynchronous calls to your models. Here’s a basic example of how to perform an inference:

```python
from vllm import AsyncLLMEngine
import asyncio


async_engine = AsyncLLMEngine()
async def main():
    response = await async_engine.infer(prompt="Hello, world!")
    print(response)

asyncio.run(main())

```


This code snippet demonstrates how to use the infer method to get predictions from the model asynchronously.




## References
1. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
2. POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference
3. Orca: A Distributed Serving System for Transformer-Based Generative Models
4. https://www.anyscale.com/blog/continuous-batching-llm-inference
5. [vLLM slides](https://docs.google.com/presentation/d/1_q_aW_ioMJWUImf1s1YM-ZhjXz8cUeL0IJvaquOYBeA/)
