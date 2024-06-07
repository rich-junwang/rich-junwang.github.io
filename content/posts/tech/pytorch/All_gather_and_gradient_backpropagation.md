---
title: "All Gather and Gradients"
date: 2023-12-11T00:18:23+08:00
lastmod: 2023-12-11T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- blog
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
In PyTorch, autograd keeps a record of data (tensors) & all executed operations (along with the resulting new tensors) in a directed acyclic graph (DAG) consisting of Function objects. In this DAG, leaves are the input tensors, roots are the output tensors. By tracing this graph from roots to leaves, we can automatically compute the gradients using the chain rule.
<p></p>
In a forward pass, autograd does two things simultaneously:

- run the requested operation to compute a resulting tensor, and
- maintain the operation’s gradient function in the DAG.
<!--   # to block the following text block to generate p tag and creating extra space with listing items.  -->
<p></p>
The backward pass kicks off when .backward() is called on the DAG root. autograd then:

- computes the gradients from each [grad_fn](https://amsword.medium.com/understanding-pytorchs-autograd-with-grad-fn-and-next-functions-b2c4836daa00),
- accumulates them in the respective tensor’s .grad attribute, and
- using the chain rule, propagates all the way to the leaf tensors.

From this, we can know that when we call functions like ` torch.distributed.all_gather`, the resulting tensors do not propagate back gradients. This can be verified with the following code snippet. 

```python
import os
import torch
from torch import nn

batch_size = 16
rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))
bs_each = batch_size // world_size
device_id = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
torch.cuda.set_device(device_id)
torch.distributed.init_process_group(
    backend='nccl',
    init_method='tcp://localhost:12345',
    rank=rank,
    world_size=world_size,
)

model = nn.Linear(1, 1, bias=False)
model.weight.data[:] = 1.
model = model.cuda()
x = torch.ones((bs_each, 1), requires_grad=True).cuda()
y = model(x)
ys = [torch.zeros_like(y) for i in range(world_size)]
torch.distributed.all_gather(ys, y)
print(y.grad_fn)
#<MmBackward object at 0x7ff10dfea500>
for x in ys:
     print(x.grad_fn)   # None
     print(x.requires_grad)  # False
```

Here we talk about how to use all_gather function in the pytorch so that we could still leverage auto_grad to help us for backpropagation. 

### Solution One
We can wrap the all_gather function and pass the context information to the gathered tensor.
```python
import torch
import torch.distributed as dist

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
```

### Solution Two
As shown below, we put the auto_grad captured tensor back to the gather tensor. In this way, this specific element on current rank will have gradient. 
```python
all_x = [torch.zeros_like(x) for _ in range(world_size)]
torch.distributed.all_gather(all_x, x)
all_x[rank] = x
```
### References
<!-- 1. https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8 -->
1. https://github.com/Spijkervet/SimCLR
2. https://github.com/princeton-nlp/SimCSE
