---
title: "Coder Training"
date: 2023-12-18T00:18:23+08:00
lastmod: 2023-12-18T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- LLM
- ML
description: "Large coder model pretraining"
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

In the code pre-training, it is often necessary to generate corresponding inserted content based on both the left context and right context. Thus, in code pretraining we have an additional task called fill-in-the-middle. 

### Data Format

With a certain probability $p$ called the FIM rate, documents are cut into three parts: prefix, middle, and suffix. For PSM format,
it arrange the text as the following format
```
<PRE> ◦ Enc(prefix) ◦ <SUF> ◦ Enc(suffix) ◦ <MID> ◦ Enc(middle)
```
`<PRE>`, `<SUF>` and `<MID>` are special sentinel tokens. 

Accordingly, SPM format we swap the order of prefix and suffix. At training time, we can jointly do both PSM and SPM training. At inference time, we can choose either as the inference format.




### References
1. [Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/pdf/2207.14255.pdf)