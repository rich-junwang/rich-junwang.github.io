---
title: "LoRA Model Fine-tuning"
date: 2023-05-05T00:18:23+08:00
lastmod: 2023-05-05T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- ML
description: "LoRA finetuning"
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

Light-weight LoRA fine-tuning seems to be the go-to option for many application where compute resources are limited. 

### LoRA

The way lora works is illustrated in the figure below. For some matrices in the transformer model, we add a parallel weighted matrix. 
The branch matrix can be decomposed into two smaller matrix. At training time, we only train these small matrices with original weights frozen. 

At inference time, we don't need to maintain separate parameters, thus we merge the LoRA weights into the original model weights.


<p align="center">
    <img alt="lora" src="images/lora.png" width="80%" height=auto/> 
    Figure 1. Lora Fine-tuning and Inference
    <br>
</p>


### LoRA Fine-tuning with Deepspeed

Huggingface PEFT package already provides easy-to-use APIs for LoRA fine-tuning. However, when we combine these with Deepspeed, we need to be careful when we merge model weights. 

Below we assume we have Deepspeed checkpoints and we want to have inference model weights with LoRA parameters merged. We have to follow the following steps:
1. Convert Zero checkpoint into a single shard fp32 checkpoint
2. Load the original model before fine-tuning.
3. Get peft config and get peft model using base model and peft config. This kind be done with PEFT API `get_peft_model`
4. Load single shard zero ckpt from step 1 into step3 model definition
5. do **merge_and_unload** and save pretrained model.  

## Reference
1. https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
2. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
