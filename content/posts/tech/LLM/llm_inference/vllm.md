---
title: "VLLM"
date: 2024-01-05T00:18:23+08:00
lastmod: 2024-01-05T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- ML
description: "VLLM"
weight:
slug: ""
draft: true # 是否为草稿
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

### PagedAttention

Traditional LLM inference has inefficient KV cache management. There are three issues:
 - reserved slots for future tokens: although the slots are used eventually, the pre-allocation leads to most of time the memory slots are wasted . Think about the end token slot is empty in the whole decoding process.  
 - Internal fragmentation due to over-provisioning for potential maximum sequence lengths,
 - External fragmentation from the memory allocator like the buddy allocator.




### References
1. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://dl.acm.org/doi/pdf/10.1145/3600006.3613165)