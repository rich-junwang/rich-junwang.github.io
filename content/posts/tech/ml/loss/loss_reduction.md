---
title: "Loss Reduction"
date: 2024-06-18T00:18:23+08:00
lastmod: 2024-07-18T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
    - ML
description: "Losses in ML"
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

The devil is in the details. Many times, when model doesn't work as expected, it's most likely there are nuances that are not taken care of in implementation. Today we talk about a common issue in LLM implementations -- loss reduction. 

For multi-turn chat mode data, the data could contain multiple roles and one training instance could have multiple sessions.
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You're a helpful assistant!"
    },
    {
      "role": "user",
      "content": "When was George Washington born?"
    },
    {
      "role": "assistant",
      "content": "George Washington was born on February 22, 1732."
    },
    {
      "role": "user",
      "content": "When he was the president?"
    },
    {
      "role": "assistant",
      "content": "George Washington served as the first President of the United States from April 30, 1789 to March 4, 1797."
    }
  ]
}
```

For SFT training, we can formulate the training data into two kinds of format:

```python
# break multi-turn chat into single response chat, e.g. for the example below, from one training instance, we can get 3 training examples
# loss is only computed for `assistant1` in first example, `assistant2` in second example, and `assistant3` for the third example
# 
|--system prompt--|--user1--|--assistant1--|-----------------------padding------------------------------|
|--system prompt--|--user1--|--assistant1--|--user2--|--assistant2--|----------------padding------------|
|--system prompt--|--user1--|--assistant1--|--user2--|--assistant2--|--user3--|--assistant3----padding--|


# keep the multi-turn chat and add loss mask, only compute loss for `assistant1`, `assistant2` and `assistant3`
# This is the same with when we just pack multiple examples into one input
|--system prompt--|--user1--|--assistant1--|--user2--|--assistant2--|--user3--|--assistant3----padding--|
```

Many algorithms utilize a sample-level loss computation strategy, wherein losses are initially averaged across tokens within each individual sample. Subsequently, these sample-level losses are aggregated across a batch of samples. This method ensures that each sample contributes equally to the overall loss calculation, thereby maintaining uniform weighting across samples.


However, this approach has an issue when response lengths vary a lot. Assume in the above two cases, we have $m$ training examples. The loss equations are as follows for the two data formats respectively


$$
Loss_{total} = \frac{1}{m} \left( \frac{loss_1}{n_1} + \frac{loss_2}{n_2} + \frac{loss_3}{n_3} \right)
$$


$$
Loss_{total} = \frac{1}{m}  \frac{loss_1 + loss_2 + loss_3}{n_1 + n_2 + n_3}
$$


For the first case, the long response tokens is equivalently down-weighted, short response loss is amplified. On the contrary, for the second case, the short response examples get down-weighted because it's overwhelmed by long responses. 

This happens not only for multi-turn chat, but also when we use gradient accumulation (as discussed [here](https://github.com/huggingface/transformers/issues/24725)), data parallel loss reduction.



It's worth to note that this is not always we want. For example, when we train reasoning model in RL, we want to increase the weight of long response and down weight the short response. If we just take the first approach, tokens within longer generation contribute less to the final averaged loss. The consequence is that (1) long gibberish generation is not punished enough (2) Good long generation is not rewarded enough. 
We need to do all token level averaging in a mini batch. This is discussed in Ref [1]. 




## Reference
1. DAPO: An Open-Source LLM Reinforcement Learning System at Scale
2. https://github.com/huggingface/transformers/issues/24725
3. https://zhuanlan.zhihu.com/p/721652210

