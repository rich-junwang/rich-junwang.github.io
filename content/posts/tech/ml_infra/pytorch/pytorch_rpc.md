---
title: "RPC in Torch"
date: 2024-01-11T00:18:23+08:00
lastmod: 2024-01-11T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- Infra
description: "PyTorch.."
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

PyTorch supports several approaches to distributed training and communication, including:

- Data Parallel (DP): A legacy approach where data is split across multiple GPUs on a single machine, and model replicas are kept synchronized manually.

- Collective Communications (c10d): A low-level library providing collective communication primitives (e.g., all-reduce) for synchronizing data across multiple devices and nodes. It supports MPI, NCCL and GLOO backend. 

- Distributed Data Parallel (DDP): The recommended method for scaling training across multiple GPUs and machines, where each process maintains its own model replica and gradient synchronization is performed efficiently using collective communication.

- Remote Procedure Call (RPC) Framework: A flexible framework enabling model parallelism by allowing different parts of a model to reside and execute on different devices or machines through asynchronous remote execution.

Data parallel is not really a distributed training tool which is discussed in our previous post, we'll not cover it here. In this post, we mainly want to talk about why Torch introduce TorchRPC. 


## Why RPC
### Conditional Communication
Without RPC (only using p2p communication in `torch.distributed` .send/.recv), when we want to send tensor y when tensor x is ready from node A to node B, both nodes need to know ahead of time exactly what communication will happen.

A has tensor x and wants to decide: "Should I send tensor y to B or not?"
With just .send() and .recv(), both A and B must call these communication functions in matching order — otherwise, they will deadlock (e.g., A is sending but B is not receiving, so both get stuck forever). The problem is B cannot "wait to see" if A wants to send, unless B already knows whether a .recv() is needed. Therefore, B must somehow know what A decided, and in simple .send/.recv world, the only way is to first send x (the condition) to B, so B can evaluate the logic too.


### Nodes Communication

RPC establishes a p2p communication without requiring `init_process_group`. For example:

```python
import multiprocessing as mp
import torch


def main(rank,world_size):
    
    torch.distributed.init_process_group(rank=0,world_size=1,backend='nccl',init_method=f'tcp://127.0.0.1:{29500+rank}')
    options = torch.distributed.rpc.TensorPipeRpcBackendOptions(init_method='tcp://127.0.0.1:30001')
    torch.distributed.rpc.init_rpc(f'worker-{rank}', rank=rank, world_size=world_size, rpc_backend_options=options)

    print(f'rank: {torch.distributed.get_rank()}',
          f' world_size: {torch.distributed.get_world_size()}',
          f' {torch.distributed.rpc.get_worker_info()}')
    torch.distributed.rpc.shutdown()


if __name__ == '__main__':
    world_size = 4
    ps = [mp.Process(None,main,args=(rank,world_size)) for rank in range(world_size)]

    for p in ps:
        p.start()
    for p in ps:
        p.join()
```




|                | P2P (.send/.recv)                      | RPC                                 |
|----------------|----------------------------------------|-------------------------------------|
| Communication  | Must be prearranged on both sides      | Triggered by sender only            |
| Flexibility    | Low, hard for conditional logic        | High, can easily send conditionally |
| Failure mode   | Deadlock if not matching               | No deadlock if only one side triggers |
| Best for       | Simple, fixed communication            | Complex, dynamic, conditional logic |



## References
1. https://medium.com/@eeyuhao/pytorch-distributed-a-bottom-up-perspective-e3159ee2c2e7