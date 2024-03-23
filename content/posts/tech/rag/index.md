---
title: "Retrieval Augmented Generation"
date: 2023-10-18T00:18:23+08:00
lastmod: 2023-10-18T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
    - ML
description: "RAG system"
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

LLMs has made remarkable progress these days, however, they still exhibit notable limitations. Among these, hallucination is one of the most seen issues. In other words, the generations from LLMs are not grounded. To this end, people are turning to retrieval augmented generation to tackle the issue. In this blog, let’s roll up our sleeves and dive deep into the retrieval augmented system. 

RAG system contains three parts: indexing, querying and generation. Indexing is the offline process which is the crucial data modeling phase. 

### Retrieval
Retrieval is the online process where the system converts user query into vector representation and retrieve relevant documents. 


#### Retrieval Evaluation Metric
Like recommender system, retrieval system commonly use the following evaluation metrics.
- Hit ratio (hit@k), 
- Normalized Discounted Cumulative Gain (nDCG), 
- Precision@k, 
- Recall@k
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)

Hit@k sometimes is also called Top-k accuracy. It is the percentage of search queries for each of which at least one item from the ground truth is returned within the top-k results. Simply put, it means % of queries get answer hit at top k retrieved passages. (Answer hit means user clicked on the doc). 




### Generation


### References
[1] [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)



<!-- [x] Effective reformulation of query for code search using crowdsourced knowledge and extra-large data analystics. -->





