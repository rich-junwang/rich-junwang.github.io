---
title: "Data Sampling in Training"
date: 2023-05-05T00:18:23+08:00
lastmod: 2023-05-05T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- distributed training
description: "Distributed data processing"
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


1. Probability proportional to size sampling (PPS sampling)
$$
q_i = \frac{p_i^t}{\sum_j p_j^t}
$$

Here $p_i$ is the number of sequences in the $i_{th}$ dataset. In this sampling, it will keep the largest dataset intact and up sample the remaining dataset to conform to the distribution in q. $t$ is the temperature. When $t$ is 0, it's uniform sampling. When $t$ is 1.0, no up-sampling will be performed.

