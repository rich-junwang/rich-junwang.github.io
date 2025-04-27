---
title: "Autograd"
date: 2024-03-11T00:18:23+08:00
lastmod: 2024-03-11T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- ML
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


In PyTorch's computation graph, there are only two types of elements: data (tensors) and operations (ops).

Operations include: addition, subtraction, multiplication, division, square root, exponentiation, trigonometric functions, and other differentiable operations.

Data has leaf nodes which are created by user and non-leaf node. The difference is after back propagation, gradient of non-leaf nodes will be released to save memory. If we want to retain non-leaf node gradient, we have to use `retain_grad`.

## Tensor 

Tensor in Pytorch has the following attributes:
1. data: stored data
2. require_grad: whether need to compute gradient. Self-defined leaf nodes usually default require_grad as False, and non-leaf nodes default as True. Neural network weights default as True. 
3. grad: grad holds the value of gradient. Each time when performing a backward computation, you need to reset (zero out) the gradients from the previous step; otherwise, the gradient values will keep accumulating. 
4. grad_fn: This is the backward function used to calculate the gradient. Leaf nodes usually have `None` for their grad_fn, and only the result nodes have a valid grad_fn, which indicates the type of gradient function.
5. is_leaf




## Gradient Computation

There are two ways to compute grad in Pytorch. 
- Backward(): used to compute grad for leaf node. 
- torch.autograd.grad() : Automatic grad computation


### Backward
Let's first take a look at `backward()` function. The definition of the `backward()` function of the `torch.autograd` is as follows

```python
torch.autograd.backward(
    tensors, 
    grad_tensors=None, 
    retain_graph=None, 
    create_graph=False, 
    grad_variables=None
)

```

Here is the meaning of the parameters here: 

> tensor: The tensor used for gradient computation. In other words, these two ways are equivalent: `torch.autograd.backward(z) == z.backward()`.
> grad_tensors: Used when computing gradients for matrices. It is also a tensor, and its shape generally needs to match the shape of the preceding tensor.
> retain_graph: Normally, after calling backward once, PyTorch will automatically destroy the computation graph. So if you want to call backward on a variable multiple times, you need to set this parameter to True.
> create_graph: When set to True, it allows the computation of higher-order gradients.
> grad_variables: According to the official documentation, "grad_variables is deprecated. Use grad_tensors instead." In other words, this parameter will likely be removed in future versions, so just use grad_tensors.

Note that here `t.backward()` is equivalent to `torch.autograd.backward(t)`.

#### Scaler Backward

By default, autograd can only compute gradient for a scaler using `backward` function. For example:

```python
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x**2+y
z.backward()
print(z, x.grad, y.grad)

# tensor(7., grad_fn=<AddBackward0>) tensor(4.) tensor(1.)

```

#### Tensor Backward

```python
x = torch.ones(2,requires_grad=True)
z = x + 2
z.backward()

# raise RuntimeError: grad can be implicitly created only for scalar outputs


x = torch.ones(2,requires_grad=True)
z = x + 2
z.backward(torch.ones_like(z))
```

We can sum `z` here to compute the grad. Or we can use the `grad_tensor` to multiply with `z` to compute the tensor. 



### Autograd

The internal nodes gradient are compute with autograd. Its interface is defined as below: 

```python
# pytorch interface
torch.autograd.grad(
    outputs, 
    inputs, 
    grad_outputs=None, 
    retain_graph=None, 
    create_graph=False, 
    only_inputs=True, 
    allow_unused=False
)
```

We can also compute the gradient for leaf node using autograd. For example,

```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x**2+y
z.backward()
print(z, x.grad, y.grad)


x = torch.tensor(2.0, requires_grad=True)
z = x**2
print(torch.autograd.grad(outputs=z, inputs=x), x.grad)

```




## Grad_fn and next_functions

How the backward computation graph works with grad_fn and next_functions? 

Essentially `grad_fn` is an objection which 
- a callable to compute current step gradient with respect to input (such as loss)
- a pointer to previous compute node `grad_fn` through `next_functions`. Think of this as a linked list.


```python
# Example from ref 1. 
torch.manual_seed(6)
x = torch.randn(4, 4, requires_grad=True)
y = torch.randn(4, 4, requires_grad=True)
z = x * y
l = z.sum()
l.backward()
print(x.grad)
print(y.grad)

# Notice that we have ops (like multiply, sum) and tensors (x, y, z, l)
# Forward
# x
#     \
#         multi  -> z  -> sum  -> l
#     /
# y

# backward
# dx
#     \
#         back_multi  <- dz  <- back_sum  <- dl
#     /
# dy

# equivalent 
torch.manual_seed(6)
x = torch.randn(4, 4, requires_grad=True)
y = torch.randn(4, 4, requires_grad=True)
z = x * y
l = z.sum()
dl = torch.tensor(1.)
back_sum = l.grad_fn
dz = back_sum(dl)
back_mul = back_sum.next_functions[0][0]
dx, dy = back_mul(dz)
back_x = back_mul.next_functions[0][0]
back_x(dx)
back_y = back_mul.next_functions[1][0]
back_y(dy)
print(x.grad)
print(y.grad)
```


Another example [3]
```python

# Notice that we have ops (like multiply, sum) and tensors (A, B, C etc)
# A
#     \
#       multi -> C  -> exp  -> D  -> sum  -> F
#     /                           /  
# B                            E 


A = torch.tensor(2., requires_grad=True)
B = torch.tensor(.5, requires_grad=True)
E = torch.tensor(1., requires_grad=True)
C = A * B
D = C.exp()
F = D + E
# tensor(3.7183, grad_fn=<AddBackward0>) 打印计算结果，可以看到F的grad_fn指向AddBackward，即产生F的运算
print(F)       

# [True, True, False, False, True, False] 打印是否为叶节点，由用户创建，且requires_grad设为True的节点为叶节点
print([x.is_leaf for x in [A, B, C, D, E, F]])


# [<AddBackward0 object at 0x7f972de8c7b8>, <ExpBackward object at 0x7f972de8c278>, <MulBackward0 object at 0x7f972de8c2b0>, None]  
# 每个变量的grad_fn指向产生其算子的backward function，叶节点的grad_fn为空
print([x.grad_fn for x in [F, D, C, A]])    

# print ((<ExpBackward object at 0x7f972de8c390>, 0), (<AccumulateGrad object at 0x7f972de8c5f8>, 0)) 
# 由于F = D + E， 因此F.grad_fn.next_functions也存在两项，分别对应于D, E两个变量，
# 每个元组中的第一项对应于相应变量的grad_fn，第二项指示相应变量是产生其op的第几个输出。
# E作为叶节点，其上没有grad_fn，但有梯度累积函数，即AccumulateGrad（由于反传时多出可能产生梯度，需要进行累加）
print(F.grad_fn.next_functions) 

# 进行梯度反传
F.backward(retain_graph=True)   
# tensor(1.3591) tensor(5.4366) tensor(1.) 算得每个变量梯度，与求导得到的相符
print(A.grad, B.grad, E.grad)   
print(C.grad, D.grad)  

```

next_functions returns a tuple, each element of which is also a tuple with two elements. The first is the previous `grad_fn` function we need to call, e.g. back_mul in the example. The second is the argument index of the previous ops in the previous output. 






## Register Hook
`register_hook` function registers a backward hook. The hook will be called every time a gradient with respect to the Tensor is computed. The hook can be registered for both tensor and ops. 

```python
import torch

def print_grad(grad):
    print(grad)
    return grad / 2

w = torch.nn.Parameter(torch.randn(2, 2))
w.register_hook(print_grad)

loss = (w - 1) ** 2
print('before backward')
loss.mean().backward()
print('after backward')
print(w.grad)



def parameter_hook(grad):
    print('parameter hook')

def operator_hook(*grads):
    print('operator hook' )

w = torch.nn.Parameter(torch.randn(2, 2))
w.register_hook(parameter_hook)

print('first')
y = w + 1
op1 = y.grad_fn
print(op1)
op1.register_hook(operator_hook)
y.sum().backward()

print('second')
z = w + 1
op2 = z.grad_fn
print(op2)
z.sum().backward()

```



## model.eval() and torch.no_grad()
One last word at `eval()` and `no_grad()`. These two are actually unrelated. During inference, both need to be used: model.eval() sets modules like BatchNorm and Dropout to evaluation mode, ensuring the correctness of the inference results, but it does not help save memory. torch.no_grad() declares that no gradients should be calculated, which does save a lot of memory and GPU memory.


## References
1. https://amsword.medium.com/understanding-pytorchs-autograd-with-grad-fn-and-next-functions-b2c4836daa00
<!-- 2. https://zhuanlan.zhihu.com/p/10091011992
3. https://zhuanlan.zhihu.com/p/321449610 -->