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

## vLLM Inference Modes
vLLM has two inference modes:
- offline batch mode: mostly for offline model evaluation, large-scale, high-throughput inference where latency is less critical 
- online serving mode: Real-time applications like chatbots or APIs where latency is important.

The interface to these two approaches are shown in the diagram below. 

<div align="center"> <img src=images/interface.png style="width: 100%; height: auto;"/> </div>


### Offline Inference
Simpler offline batch inference example:
```python
# batch prompts
prompts = ["Hello, my name is",
           "The president of the United States is",
           "The capital of France is",
           "The future of AI is",]

# sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# load model
llm = LLM(model="facebook/opt-125m")

# Inference
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

A data parallel (SPMD style) offline inference example.

```python
import os

from vllm import LLM, SamplingParams
from vllm.utils import get_open_port

GPUs_per_dp_rank = 2
DP_size = 2


def main(dp_size, dp_rank, dp_master_ip, dp_master_port, GPUs_per_dp_rank):
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    # set devices for each dp_rank
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(i) for i in range(dp_rank * GPUs_per_dp_rank, (dp_rank + 1) *
                              GPUs_per_dp_rank))

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # with DP, each rank should process different prompts.
    # usually all the DP ranks process a full dataset,
    # and each rank processes a different part of the dataset.
    promts_per_rank = len(prompts) // dp_size
    start = dp_rank * promts_per_rank
    end = start + promts_per_rank
    prompts = prompts[start:end]
    if len(prompts) == 0:
        # if any rank has no prompts to process,
        # we need to set a placeholder prompt
        prompts = ["Placeholder"]
    print(f"DP rank {dp_rank} needs to process {len(prompts)} prompts")

    # Create a sampling params object.
    # since we are doing data parallel, every rank can have different
    # sampling params. here we set different max_tokens for different
    # ranks for demonstration.
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=16 * (dp_rank + 1))

    # Create an LLM.
    llm = LLM(model="ibm-research/PowerMoE-3b",
              tensor_parallel_size=GPUs_per_dp_rank,
              enforce_eager=True,
              enable_expert_parallel=True)
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"DP rank {dp_rank}, Prompt: {prompt!r}, "
              f"Generated text: {generated_text!r}")


if __name__ == "__main__":
    from multiprocessing import Process
    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()
    procs = []
    for i in range(DP_size):
        proc = Process(target=main,
                       args=(DP_size, i, dp_master_ip, dp_master_port,
                             GPUs_per_dp_rank))
        proc.start()
        procs.append(proc)
    exit_code = 0
    for proc in procs:
        proc.join()
        if proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)
```


### API Server
The API server utilizes Uvicorn to deploy FastAPI application. 
```bash
# Server
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf

# Client：(compatible with openai)
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b-hf",
        "prompt": "San Francisco is a",
        "max_tokens": 256,
        "temperature": 0
    }'
```

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


## Chunked Prefill

By default, vLLM scheduler prioritizes prefills and doesn’t batch prefill and decode to the same batch. Chunked prefill [7] allows to chunk large prefills into smaller chunks and batch them together with decode requests. 

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
6. https://github.com/vllm-project/vllm/pull/12071
7. SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills




<!-- 1. https://zhuanlan.zhihu.com/p/691045737
2. https://docs.google.com/presentation/d/1QL-XPFXiFpDBh86DbEegFXBXFXjix4v032GhShbKf3s -->
