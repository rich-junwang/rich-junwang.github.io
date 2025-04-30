---
title: "Entropy Collapsing in RL Training"
date: 2024-01-05T00:18:23+08:00
lastmod: 2024-01-05T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- ML
description: "Decoding"
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

## Entropy Collapsing

It's common to see entropy collapsing phenomenon in GRPO based RL model training: the entropy of token generation of policy model decreases dramatically as training progresses. 

Before diving deep into the topic, let's first take a look at the entropy and varentropy of LLM generation process. Entropy measures how confident the model is when generating a specific token. It's calculated at each generation step. Low entropy indicates that the model is certain about next token, i.e. the probabilities are concentrated on a few tokens.

$$
H(X) = -\sum_{x \in \mathcal{X}} P(x) \log P(x)
$$

Another concept is varentropy. Variance of (− log p(X)) of the discrete random variable X is called
the varentropy. Varentropy is a measure of how the uncertainty varies across different tokens in generation. Mathematically the varentropy is defined as

$$
\begin{aligned}
V(X) &= \text{Var}(-\log P(X))  \\\
&= \sum_{x \in \mathcal{X}} P(x) \left(-\log_2 P(x) - H(X)\right)^2 \\\
&= \mathbb{E}[(-\log P(X))^2] - (H(X))^2
\end{aligned}
$$

Low varentropy means the model’s uncertainty is consistent across tokens whereas the high varentropy means that model's uncertainty varies significantly in generated tokens. 

The computation of entropy and varentropy in python is implemented below.
```python
import torch
from typing import Optional, Tuple

def calculate_entropy_and_varentropy(
    probs: torch.Tensor,
    log_probs: Optional[torch.Tensor] = None,
    eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute entropy and varentropy (variance of surprise) of a discrete probability distribution.
    
    Args:
        probs (torch.Tensor): Probability tensor of shape (..., num_classes), where the last dim sums to 1.
        log_probs (Optional[torch.Tensor]): Optional precomputed log2 probabilities (same shape as probs).
        eps (float): Small constant for numerical stability to avoid log(0).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple (entropy, varentropy), each of shape (...) matching the batch dimensions.
    """
    # Ensure numerical stability
    safe_probs = torch.clamp(probs, min=eps)

    # Compute log2 probabilities if not provided
    if log_probs is None:
        log_probs = torch.log2(safe_probs)

    # Compute entropy: H(X) = -Σ p(x) log2 p(x)
    entropy = -torch.sum(safe_probs * log_probs, dim=-1)

    # Compute varentropy: V(X) = Σ p(x) (−log2 p(x) − H(X))²
    surprise = -log_probs
    mean_surprise = entropy.unsqueeze(-1)  # shape (..., 1) for broadcasting
    squared_deviation = (surprise - mean_surprise) ** 2
    varentropy = torch.sum(safe_probs * squared_deviation, dim=-1)

    return entropy, varentropy

```

As is shown in ref 4, LLMs cannot reason if we only consider the greedy decoding path. In other words, reasoning is achieved through a less certain decoding path. In the DAPO paper, the solution is to increase the clip upper bound of importance sampling ratio. Through adjusting the $\epsilon$, we effectively increase the possibility of choosing low-probability tokens. 

As we figure out that the approach to prevent entropy collapsing is to increase the entropy of generated tokens, we can implement entropy based sampling approach in RL training. Ref 1 has provide a good base solution for this kind of entropy based sampling.








### References
1. https://github.com/xjdr-alt/entropix
2. DAPO: An Open-Source LLM Reinforcement Learning System at Scale
3. https://southbridge-research.notion.site/Entropixplained-11e5fec70db18022b083d7d7b0e93505
4. Chain-of-Thought Reasoning without Prompting