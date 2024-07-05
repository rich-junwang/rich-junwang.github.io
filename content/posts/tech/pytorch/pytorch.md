---
title: "Pytorch Multiple-GPU Training"
date: 2022-06-11T00:18:23+08:00
lastmod: 2023-04-11T00:18:23+08:00
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

Using PyTorch for NN model training on single GPU is simple and easy. However, when it comes to multiple GPU training, there could be various of issues. In this blog, I'll summarize all kinds of issues I ran into during model training/evaluation.


### Gradient Accumulation
Gradient accumulation is a way to virtually increase the batch size during training. In gradient accumulation, `N` batches go through the forward path and backward path, and each time, the gradient is computed and accumulated (usually summed or averaged), but model parameters are not updated. Model parameters are updated after iterate through all `N` batches. The logic is as follows:
```
for step, oneBatch in enumerate(dataloader):
   ... 
   ypred = model(oneBatch)
   loss = loss_func(ytrue, ypred)
   loss.backward() # release all activations memory
   if step % accumulation_step == 0: 
      # update weights every accumulation_step steps
      loss.step() 
      loss.zero_grad()
```
In order to backpropagate, all the hidden activations must be stored until we call loss.backward(). In contrast, if we only add losses together (accumulating losses), all the activation memory won't be released, so we can't save memory. 
<!-- ```
totalLoss = 0
for step, eachBatch in enumerate(dataloader):
   ... 
   loss = loss_func(ytrue, ypred)
   totalLoss += loss # accumulate the loss
   if step % 5 ==0: 
      totalLoss.backward()
      totalLoss.step()
      totalLoss.zero_grad()
      totalLoss=0

``` -->

### PyTorch Inference
At inference time, we call `model.eval()` so that model wouldn't calculate the gradient. It would still be beneficial to wrap the code block with `with torch.no_grad()`. The reason is it seems PyTorch creates grad buffer for the input tensors created in the computation graph. With this no_grad wrapper, it could free more spaces. 


### PyTorch DataParallel
DataParallel is very easy to use, we just wrap the model with `DataParallel()` wrapper. The input should be splittable on dim 0. Caveat here normally when we directly feed output of tokenizer into model, e.g. using `tokenizer(input)` as model input, sthis will lead to unsplittable tensors.  

The issue with `DataParallel` is unbalanced GPU usage. The input is splitted to each GPU, and gathered on default GPU (usually cuda:0). Thus, the default GPU has much larger memory load. 

A way to solve this issue is to wrap your model and make sure most ops are done on each GPU. Model only return a small tensor. 
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


input_size = 5
output_size = 2
batch_size = 30
data_size = 100000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)

for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())

```

Always make sure that the batch size is divisible by 8. If not, we can do this simple trick. This is helpful when we don't use dataloadder and sampler.
```python
# get smaller number that is greater than x and is multiples of 8
def roundup(x):
   return (x + 7) & (-8)

if len(input_batch) < batch_size:
   new_batch_size = roundup(len(input_batch))
   input_batch += [input_batch[-1]] * (new_batch_size - len(input_batch))
``` 

### Install Apex
Sometimes to use the latest distributed training feature, we have to install Apex. As Apex is closely coupled with Cuda, we need to follow the next few steps to correctlly install apex.
- Find out the Cuda version used in the system. 
```
python -c "import torch; print(torch.version.cuda)"
```

- Install from source
```
git clone https://github.com/NVIDIA/apex
cd apex
CUDA_HOME=/usr/local/cuda-{your-version-here}/ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


### Commonly Used Pytorch Tricks
Distributed training is error-prone, so effective ways of debugging is needed. Here I document some of these commands
```
# print the whole tensor
torch.set_printoptions(profile="full")
torch.set_printoptions(linewidth=16000)
```

Dataloader sometimes can be buggy, when there are errors related to dataloader, a good practice is to disable the worker number and disable prefetching.


### Launch Distributed Run
```bash
python3 -m torch.distributed.run --nnodes=2 --nproc_per_node 8 --node_rank=${NODE_RANK} --master_port=1234 --master_addr=xxx train.py args..
```


### Pytorch and Numpy Advanced Indexing
When selection object is sequence object, ndarray/tensor, it will trigger advanced indexing. To understand how it works, we start from simple.
```python
x = np.arange(12).reshape(4,3)
print(x)

#output
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
```

(1) Specify integer arrays in each dimension where every element in the array represents a number of indices into that dimension. In the example below, we select (0, 0), (1, 1), (2, 0) elements from the above array. `x` has two dimensions so we have two arrays to specify the indices in each dimension.
```
y = x[[0,1,2],  [0,1,0]]
print(y) 
#[0 4 6]
```

(2) The above way of indexing only renders single dimension result. We can use multi-dimension array to get multi-dimension output. Below is one of these examples. This is to select [(0, 0), (0, 2)], [(3, 0), (3, 2)] elements. Note that in each dimension we still only select one index, like 0 from row-dim, and 0 from col-dim. 
```
rows = np.array([[0,0],[3,3]]) 
cols = np.array([[0,2],[0,2]])
y = x[rows,cols]  
print (y)

# output
[[ 0  2]
 [ 9 11]]
```

### Loading a pretrained checkpoint
A lot of times when we save a checkpoint of a pretrained model, we also save the trainer (or model state) information. This means when we load model checkpoint again, model will already have a preallocated device. When we use the same number of GPU to continue training, it will work as expected. However, the issue will arise when we have different number of GPUs for two runs. Let's say, we first trained model on a single GPU, then we want to use multiple GPU to continue the training. When we move model to multiple GPU, there will be something weird. For instance, on GPU 0, you might see multiple process (normally one process per GPU). Or in other cases, you can see GPU 0 has much higher memory usage than other GPUs. 

Solution: when we load model, we only load parameters and strip all state information. This might be tricky sometimes. The simplest way to solve this issue is to wrap the command with with PyTorch distributed data parallel. 
```
python3 -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 my_script.py my_config_file 
```