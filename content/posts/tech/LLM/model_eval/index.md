---
title: "Model Evaluation"
date: 2023-10-18T00:18:23+08:00
lastmod: 2023-10-18T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
    - ML
description: "Model Evaluation"
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

## Model Evaluation Robustness
LLMs performance is sensitive to evaluation details. One of my previous co-workers's work shows that for popular multiple choice question
benchmarks (e.g. MMLU) minor perturbations to the benchmark, such as changing the order of choices or the method of answer selection, result in changes in rankings up to 8 positions.


## PAL for Math Reasoning
In [PAL](https://arxiv.org/pdf/2211.10435.pdf) paper, the authors found that solving mathematical problems using external tools (Python interpreter) could greatly boost math reasoning performance.


### References
1. [When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards](https://arxiv.org/abs/2402.01781)
2. [Increasing Probability Mass on Answer Choices Does Not Always Improve Accuracy](https://arxiv.org/pdf/2305.14596.pdf)


<!-- https://github.com/deepseek-ai/DeepSeek-Coder/tree/main/Evaluation/PAL-Math -->