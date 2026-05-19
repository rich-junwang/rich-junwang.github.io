---
title: "JAX"
date: 2024-11-18T00:18:23+08:00
lastmod: 2024-06-18T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
    - ML
description: "JAX"
weight:
slug: ""
draft: False # 是否为草稿
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



## GPU vs TPU


| Layer                       | NVIDIA GPU Ecosystem       | Google TPU Ecosystem         | Function                                             |
| --------------------------- | -------------------------- | ---------------------------- | ---------------------------------------------------- |
| User Framework              | PyTorch, JAX, TensorFlow   | JAX, TensorFlow, PyTorch-XLA | Model definition and training APIs                   |
| Graph Compiler              | TorchInductor, XLA/OpenXLA | XLA/OpenXLA                  | Graph capture, fusion, scheduling, optimization      |
| Manual Kernel Layer         | CUDA C++, Triton           | Pallas                       | Human-written custom kernels                         |
| Intermediate Representation | MLIR, LLVM IR, PTX         | MLIR, StableHLO              | Hardware-independent and hardware-specific IRs       |
| Backend Codegen             | CUDA backend, ptxas        | TPU backend, Mosaic          | Lowering IR into executable accelerator instructions |
| Runtime / Libraries         | cuBLAS, cuDNN, NCCL        | LibTPU, XLA runtime          | Optimized kernels and distributed execution          |


The TPU software stack is converging toward compiler-centric design, while the NVIDIA stack historically centered around explicit kernel programming.

Historically:
- NVIDIA: Program the machine directly.
- TPU: Describe computation graphs; compiler handles execution.

But today the stacks are converging:
- GPUs increasingly use compiler fusion (Inductor/XLA/Triton)
- TPUs increasingly expose lower-level control (Pallas)

## Why JIT

It enables dynamic optimizations based on the specific data, shapes, and hardware encountered during execution.


## JAX

JAX is differentiable Numpy that runs on accelerators, and relies on a purely functional programming paradigm. It is a powerful autodifferentiation library, evolved from autograd.

In PyTorch, the code is typically organized in classes. In JAX, code is organized as functions instead. JAX demands pure functions (functions with no side effects). 



Jax characteristics:

- Functional programming style
- Heavy compiler usage
- Very fast on accelerators
- Requires pure functions
- Less object-oriented than PyTorch





### JAX Pipeline

Tracing is what emits the jaxpr, while Vmap and Grad are transformations that rewrite that jaxpr before it ever reaches the compiler. JIT is the final wrapper that sends that rewritten jaxpr to the factory (XLA). Here is the exact mapping of JAX mechanisms to the pipeline steps:


| Pipeline Step | JAX Mechanism | What's Actually Happening? |
| --- | --- | --- |
| 1. Definition | Standard Python | You write your `def func(x):`. |
| 2. Tracing | jax.make_jaxpr | JAX runs the code with Tracers to see the math skeleton. |
| 3. Transformation | vmap, grad, pmap | JAX takes that math skeleton and rewrites it (e.g., adds logic for gradients or batching). |
| 4. Compilation | JIT (XLA) | JAX hands the final, rewritten skeleton to XLA to turn into high-speed machine code. |
| 5. Execution | The Runtime | The compiled executable runs on your GPU/TPU. |




#### 1. Transformations: vmap & grad

Vmap and grad are the rewriters. Think of these as JAX-to-JAX translators. This is logic transformation. 

grad: It doesn't run your function. It traces your function, looks at the jaxpr, and applies the rules of calculus to write a new jaxpr that calculates the derivative.

vmap: It traces your scalar function and lifts every operation. If the jaxpr said multiply these two numbers, vmap rewrites it to say multiply these two batches of numbers.  It intercepts the tracing process. When it sees an operation like add(a, b), it says "Wait, actually perform add across this whole axis."

#### 2. Compilation: JIT 

JIT is the optimizer. JIT is the bridge between JAX and hardware. It's the hardware optimization.
It takes the jaxpr (potentially already transformed by grad or vmap) and translates it into HLO (High-Level Operations). This is what XLA understands. XLA then looks at your specific GPU architecture and decides exactly how to fuse the math to make it run as fast as possible i.e. fuses operations together into kernels.


1. vmap makes it work on batches.
2. grad makes it return the slope of those batches.
3. jit takes that complex batch-gradient logic and bakes it into a single, optimized super-kernel for the GPU.




```python
import numpy as np

x = np.array([1, 2, 3])
# function with side effect
def in_place_modify(x):
  x[0] = 123
  return None

in_place_modify(x)
x
# array([123,   2,   3])


x = np.array([1, 2, 3])
# function without side effect
def pure_modify(x):
    new_x = x.copy()      # or x.at[0].set(123) in JAX
    new_x[0] = 123
    return new_x          # ← everything is explicit

y = pure_modify(x)
print(x)   # still [1, 2, 3]  ← original unchanged
print(y)   # [123, 2, 3]
```



## JAX Best Practices

JAX's transformations (jit, grad, vmap, pmap, etc.) assume your functions are pure:

- Same input → always same output
- No mutation of arrays
- No dependence on global state

If you mutate arrays in place, these transformations can break or give wrong results, because JAX sometimes re-runs, reorders, or traces your function in unexpected ways.

Rule of thumb in JAX:
- Never modify arrays with arr[i] = value.
- Use functional style instead (arr.at[i].set(value)).


Best practices in JAX

- Always put performance-critical code under jax.jit.
- Prefer whole-array operations and functional updates.
- Use jax.lax primitives or loops (lax.scan, lax.fori_loop) when you need iterative updates — these are highly optimized.
- For very large models/training loops, people use libraries like Equinox, Flax, or Optax that handle this pattern cleanly.



## OpenXLA

With the growing complexity of ML tasks execution across hardware and frameworks, the open community has come up with the OpenXLA project. 

PJRT (Portable JAX Runtime) is a runtime that executes JAX programs using XLA (Accelerated Linear Algebra). PJRT is very flexible: you can run your programs on CPUs, GPUs, or even TPUs without having to rewrite your code.





## References
<!-- 1. https://zhuanlan.zhihu.com/p/672327290 -->
<!-- 2. https://github.com/OpenRL-Lab/Ray_Tutorial/ -->
<!-- 1. https://rise.cs.berkeley.edu/blog/ray-tips-for-first-time-users/
2. [Ray: A Distributed Framework for Emerging AI Applications](https://arxiv.org/abs/1712.05889)
3. https://github.com/dmatrix/ray-core-tutorial
4. [Ray Whitepaper](https://docs.google.com/document/d/1tBw9A4j62ruI5omIJbMxly-la5w4q_TjyJgJL_jN2fI/preview?tab=t.0#heading=h.iyrm5j2gcdoq) -->
1. https://johnwlambert.github.io/jax-tutorial/
2. https://basicv8vc.github.io/posts/jax-tutorials-for-pytorchers-3/
3. https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html
4. Deep learning with JAX
