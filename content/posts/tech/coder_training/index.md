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

### Training Data Format

With a certain probability $p$ called the FIM rate, documents are cut into three parts: prefix, middle, and suffix. For PSM format,
it arrange the text as the following format
```
<PRE> ◦ Enc(prefix) ◦ <SUF> ◦ Enc(suffix) ◦ <MID> ◦ Enc(middle)
```
`<PRE>`, `<SUF>` and `<MID>` are special sentinel tokens. 

Accordingly, SPM format we swap the order of prefix and suffix. At training time, we can jointly do both PSM and SPM training. At inference time, we can choose either as the inference format.

FIM style training gives us the opportunity to explore more training paradigms with code data. For instance, constructing training data, we can parse the code into an abstract syntax tree (AST) and randomly select a complete node to construct a FIM task. The rationale is simple. Through this training, model could learn to predict from top-down or bottom-up. This is exactly what is needed for reasoning.

Another benefit is that model's predictions are more complete, with the generated code having a full hierarchical structure.


### Build Training Data
Given the fact that code data is mostly from github which already comes with some meta information. We could leverage these meta info to build high quality training dataset. Here are the 7 steps to build AIxcoder training data. 

1. Raw Data Selection
    - Exclude projects under copyleft licenses.
    - Deduplicate projects gathered from various code hosting platforms and open-source datasets
2. Project-Level Comprehensive Ranking
    - Calculate project metrics, including the number of Stars, Git Commit counts, and the quantity of Test files.
    - Exclude the lowest 10% of data based on a comprehensive score.
3. Code File-Level Filtering
    - Remove automatically generated code.
    - Employ near-deduplication for redundancy removal.
4. Sensitive Information Removal
    - Use named entity recognition models to identify and delete sensitive information such as names, IP addresses, account passwords, and URLs.
5. Commented Code
    - Randomly deleting large sections of commented code
6. Syntax Analysis
    - Delete code with syntax parsing errors or syntactical errors in the top fifty languages.
7. Static Analysis
    - Utilize static analysis tools to scan for and locate 161 types of Bugs affecting code reliability and maintainability, as well as 197 types of vulnerabilities impacting code security.




### References
1. [Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/pdf/2207.14255.pdf)
2. [DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence](https://arxiv.org/abs/2406.11931)
3. https://github.com/aixcoder-plugin/aiXcoder-7B