---
title: "Flash Attention"
date: 2023-06-18T00:18:23+08:00
lastmod: 2023-07-18T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
    - LLM 
    - ML
description: "Large language model pretraining"
weight:
slug: ""
draft: false # 是否为草稿
comments: true
reward: true # 打赏
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

In order to understand how flash attention and its variants help improve compute efficiency of modern LLMs training, we first have to dive deep into GPU compute model and its memory hierarchy. 

## GPU Compute Model and Memory Hierarchy
The Figure 1 here shows the high level compute model and memory in GPU. We can see that there are three types of memory affect GPU computation. CPU memory (data loading etc), GPU high bandwidth memory (the gpu memory we usually mentioned), and GPU caches (SRAM). These memories are of different size and bandwidth (read speed). Figure 2 shows the hierarcy of GPU memory in A100. 
<p align="center">
    <img alt="gpu memory" src="images/gpu_mem.png" width="100%" height=auto/> 
    <br>
    <em>GPU memory</em>
    <br>
</p>

<p align="center">
    <img alt="gpu memory hierarchy" src="images/gpu_mem_hierarchy.png" width="100%" height=auto/> 
    <br>
    <em>GPU memory hierarchy</em>
    <br>
</p>

For each computation, there are three steps of operation
- Read op — Move tensor from HBM to SRAM
- Compute op - Perform compute intensive task on SRAM
- write op - move tensor back from SRAM to HBM