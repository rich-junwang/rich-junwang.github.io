---
title: "Embedding"
date: 2023-7-05T00:18:23-08:00
lastmod: 2023-07-05T00:18:23-08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- blog
description: "Training Embedding Model"
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




### Consistency-based filter

Consistency-based filter To further improve data quality and make training costs manageable, we
propose a consistency-based data filtering technique: a model is first trained on the 1.3B noisy text
pairs, and then used to rank each pair against a pool of 1 million random passages. A text pair is kept
only if it falls in the top-k ranked lists. In other words, the model’s prediction should be consistent
with the training labels. Here we set k = 2 based on manual inspection of data quality. After this
step, we end up with ∼ 270M text pairs for contrastive pre-training.



### References
1. [Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/pdf/2212.03533)