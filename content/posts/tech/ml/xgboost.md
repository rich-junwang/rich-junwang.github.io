---
title: "XGBoost"
date: 2023-03-05T00:18:23+08:00
lastmod: 2023-03-05T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- ML
description: "xgboost"
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
---

Some parameters for using xgboost for simple classification tasks.

the ratio of features used (i.e. columns used); colsample_bytree. Lower ratios avoid over-fitting.
the ratio of the training instances used (i.e. rows used); subsample. Lower ratios avoid over-fitting.
the maximum depth of a tree; max_depth. Lower values avoid over-fitting.
the minimum loss reduction required to make a further split; gamma. Larger values avoid over-fitting.
the learning rate of our GBM (i.e. how much we update our prediction with each successive tree); eta. Lower values avoid over-fitting.
the minimum sum of instance weight needed in a leaf, in certain applications this relates directly to the minimum number of instances needed in a node; min_child_weight. Larger values avoid over-fitting.

## References
<!-- 1. https://stats.stackexchange.com/questions/443259/how-to-avoid-overfitting-in-xgboost-model -->
