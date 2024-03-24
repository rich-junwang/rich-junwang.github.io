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
- Normalized Discounted Cumulative Gain (NDCG), 
- Precision@k, 
- Recall@k
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)

Hit@k sometimes is also called Top-k accuracy. It is the percentage of search queries for each of which at least one item from the ground truth is returned within the top-k results. Simply put, it means % of queries get answer hit at top k retrieved passages. (Answer hit means user clicked on the doc). This number is meaningful when there are multiple test cases.


Normalized Discounted Cumulative Gain (NDCG) is popular method for measuring the quality of a set of search results. It asserts the following:
- Very relevant results are more useful than somewhat relevant results which are more useful than irrelevant results (cumulative gain)
- Relevant results are more useful when they appear earlier in the set of results (discounting).
- The result of the ranking should be irrelevant to the query performed (normalization).
Cumulative Gain (CG) is the predecessor of DCG and does not include the position of a result in the consideration of the usefulness of a result set. In this way, it is the sum of the graded relevance values of all results in a search result list. Suppose you were presented with a set of search results for a query and asked to rank each result. 

Given a true relevance score $R_i$ (real value) for every item, there exist several ways to define a gain. One of the most popular is:
$$
G_i = 2^{R_i} - 1
$$

where $i$ is the rank position/index of the item. The relevance score can be defined by ourselves, for instance, we can say: 0 => Not relevant 1 => Near relevant 2 => Relevant. In practical use case, it's defined as a binary value 0 and 1 where 0 is irrelevant and 1 is there is relevance.

The cumulative gain is then defined as:
$$
CG@k = \sum_{i=1}^k G_i
$$

To achieve the Discounted cumulative gain (DCG) we must discount results that appear lower. The premise of DCG is that highly relevant documents appearing lower in a search result list should be penalized as the graded relevance value is reduced logarithmically proportional to the position of the result. Thus $DCG$ is:
$$
DCG@k = \sum_{i=1}^k\frac{2^{R_i} - 1}{log_2{(i + 1)}}
$$

As DCG either goes up with $k$ or it stays the same, the queries that return larger result sets will always have higher DCG scores than queries that return small result sets. To make comparison across queries fairer is we want to normalize the DCG score by the maximum possible DCG for different $k$.
$$
NDCG@k = \frac{DCG@k}{IDCG@k}
$$
where $IDCG@k$ is the best $DCG$ we can get at position $k$. 





### Generation


### References
[1] [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) <br>
[2] [Evaluation Metrics for Ranking problems: Introduction and Examples](https://queirozf.com/entries/ evaluation-metrics-for-ranking-problems-introduction-and-examples) <br>



<!-- [x] Effective reformulation of query for code search using crowdsourced knowledge and extra-large data analystics. -->





