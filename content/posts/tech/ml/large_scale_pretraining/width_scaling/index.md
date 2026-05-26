---
title: "Scaling LLMs in Width"
date: 2025-12-05T00:18:23+08:00
lastmod: 2025-12-05T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- LLM
description: "Architecture"
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

## Scaling Width

There are a couple of dimensions that we can scale to enhance LLMs capabilities. Last time, we talked about depth dimension scaling. This time let's talk about width dimension scaling.  We always have a serious bottleneck in scaling: better models increasingly require exponentially larger computational budgets. The paper Virtual Width Networks (VWN) proposes a surprisingly elegant idea:

> What if neural networks could gain the benefits of becoming wider without paying the full computational cost that wider models normally require?

The authors argue that this may be possible by separating two things that have traditionally been tightly linked:

1. representation capacity (how much information the model can organize internally), and
2. compute capacity (how much expensive computation the model performs).

That separation is the central idea behind Virtual Width Networks.



## The Core Problem: Width Is Powerful — But Expensive

Transformer models contain hidden representations — vectors that carry information through the network. One of the most important design choices is the model’s width, usually represented by the hidden dimension (d). Larger width generally improves performance because the model has a richer internal representation space. A wider model can:

* separate concepts more cleanly,
* reduce interference between features,
* improve optimization,
* and represent more complex relationships.

But there is a major downside. Many transformer operations scale roughly quadratically with width:

$$
\text{Compute} \propto d^2
$$

That means doubling width can increase compute costs by roughly four times. So researchers face a painful tradeoff:

| Goal           | Consequence              |
| -------------- | ------------------------ |
| Increase width | Better representations   |
| Increase width | Much higher compute cost |

This is the bottleneck VWN is trying to solve.



## The Big Idea: Virtual Width

The key insight of the paper is subtle but important: The model may not need to perform all computation in the large representation space. Instead, VWN attempts to keep the expensive backbone computations relatively small, while allowing the model to behave as though it has a much larger internal width. In other words:

| Traditional Transformer    | Virtual Width Network                       |
| -------------------------- | ------------------------------------------- |
| Real width = Compute width | Representation width > Compute width        |
| Wider = Expensive          | Wider representations with lower added cost |


The paper calls this larger representation space virtual width. Here is an intuitive analogy. Imagine there are two students. Student A has a single notebook for every class. Math, biology, history, and chemistry notes are all mixed together. Student B has separate notebooks organized by topic. Even if both students think at the same speed, Student B can organize information more effectively and avoid conceptual interference. The authors argue that wider neural representations behave similarly. More representational space helps the network organize information more cleanly. Virtual Width Networks attempt to provide many “notebooks” without paying the full computational cost of processing all of them directly.



## Core Idea
In a standard Transformer, embedding dimension = hidden dimension = *D*. Widening everything to *kD* costs ~*k²* more compute/parameters (quadratic).

VWN:
- Makes input embeddings and intermediate "virtual" states** much wider (D' = r × D).
- Keeps the Transformer backbone (Attention + FFN) at the original width D.
- Uses lightweight Generalized Hyper-Connections (GHC) to shuttle information between the wide virtual space and the narrow backbone.

This achieves virtual widening by factor *r* (e.g., 8×) with only small overhead.

### Key Components

#### 1. Over-Width Embedding
- Start with a normal token embedding, then expand it to width **D' = rD**.
- For large *r*, they use a linear projection: `E_wide = W_expand * E_base`.
- The model works with these **Over-Width Hidden States** (`h' ∈ ℝ^{D'}`) throughout.

#### 2. Generalized Hyper-Connections (GHC) — The Core Mechanism
This is how they efficiently move between wide virtual states and narrow backbone states.

They partition the hidden states:
- Divide the backbone width *D* into **m** segments (each of size *D/m*).
- Divide the virtual width *D'* into **n** segments (each of size *D'/n*).
- The **virtual width factor** is **r = n / m**.

**GHC** uses two small routing matrices per layer *l*:
- **A^l** (size roughly *n × (m+n)*) — mixes virtual states.
- **B^l** (size *m × n*) — writes backbone outputs back into virtual slots.

The update rule (simplified):

```python
H'_l = B^l^T * (T^l (A^l^T * H'_{l-1})) + Â^l^T * H'_{l-1}
```

Where:
- `T^l` = Transformer block (Attention or FFN) operating at normal width *D*.
- `H'` = Over-width hidden states, reshaped into `(n, D'/n)` matrix of "slots".
- `Â` is the carry/forget part of A.

**Dynamic GHC (DGHC)** makes A and B input-dependent (like a very lightweight linear attention over depth):
- Static initialization (cyclic for B, block-identity for A).
- Dynamic part generated via small projection + tanh + scaling.

This acts like learned linear attention along the depth axis with a compressed "depth KV cache".

#### 3. Reduce Operator at the End
At the final layer:
- Apply GroupNorm on the wide states (group size = original *D*).
- Linear projection `W_reduce: D' → D` to bring it back to standard width before the LM head.

#### 4. Multi-Token Prediction (MTP) Synergy
- They combine VWN with multi-token prediction objectives.
- For the MTP head, they use block-wise mixing (small linears per segment) to avoid quadratic cost in the wide space.

### Parameter Choices (m, n, r)
- *m*: Controls partitioning granularity (compression per "slot").
- *n*: Controls number of virtual slots.
- *r = n/m*: Virtual widening factor.

Examples:
- **1.5×**: `(m=2, n=3)`
- **8×**: `(m=8, n=64)`

Larger *m* remembers more layers (at lower fidelity per layer). Larger *r* gives more total virtual capacity.

### Cost
- **Compute**: Very small overhead (mainly extra normalizations + small matrix multiplies for A/B). They report ~few % extra FLOPs.
- **Memory**: Minor activation overhead, mitigated by recomputation.
- Main cost is wider embedding table and final reduce, but these are cheap compared to quadratic backbone widening.

### Empirical Scaling
They found an approximately log-linear relationship:
- Loss reduction ≈ *c × log₂(r)*
- Each doubling of virtual width gives consistent (though modest) gains.
- Gains **increase with scale** and training time.
- 8× virtual width gave **2.5× token efficiency** on next-token and **3.5×** on next-2-token prediction on a 3.3B MoE model.


VWN implements virtual width by:
1. Widening only the embeddings + maintaining wide "virtual hidden states".
2. Using GHC (small static+dynamic routing matrices) as a cheap bridge to a normal-width Transformer backbone.
3. Treating the depth axis as a compressible memory with learned carry/write operations.

It's essentially a more general and powerful evolution of ideas like Hyper-Connections and AltUp, reframed as "virtual width scaling" with strong empirical scaling laws.






## Why Width Helps Neural Networks

The paper builds on an increasingly important observation in deep learning: Wider representations often improve optimization.

Why? Researchers believe wider networks may:

* reduce feature collisions,
* improve gradient flow,
* create smoother optimization landscapes,
* and allow more disentangled internal representations.

In simpler terms: narrow models force many concepts to compete for limited representational space, while wider models allow concepts to occupy separate regions. That separation can make learning substantially easier. Historically, width has been underexplored because widening transformers becomes extremely expensive. VWN attempts to unlock width as a scalable training dimension.

## What the Paper Reports

The paper reports several notable improvements:

* approximately 2× faster optimization for next-token prediction,
* approximately 3× faster optimization for next-2-token prediction,
* stronger gains later in training,
* and approximately log-linear scaling between virtual width and loss improvements.

The authors observe behavior resembling:

$$
\Delta \text{Loss} \propto \log(\text{Virtual Width})
$$

This suggests that increasing virtual width continues to help, although with diminishing returns.

That pattern resembles many known neural scaling laws.



## Why This Is Potentially Important

Training frontier AI models is extraordinarily expensive.

A significant portion of the cost comes from:

* slow convergence,
* massive token budgets,
* and inefficient scaling.

If Virtual Width Networks genuinely allow models to reach lower loss faster, several important consequences follow:

### 1. Lower Training Costs

Faster optimization means fewer GPU-hours are required.

That reduces the cost of training large models.

### 2. Faster Research Iteration

Researchers can experiment more rapidly when training runs complete sooner.

### 3. New Scaling Pathways

The paper introduces a possible new scaling axis:

* not just deeper models,
* not just larger parameter counts,
* but larger *effective representation spaces*.

This could become an important direction for future AI architectures.


## Open Questions and Skepticism

Like many promising architectural ideas, VWN still raises important questions.

### Does It Scale to Frontier Models?

Many methods perform well at medium scale but fail to hold up at trillion-token training regimes.

### Does Faster Optimization Improve Final Capability?

Improved training curves do not always translate into stronger final reasoning ability.

### Is the Benefit Primarily Optimization?

The gains could stem from:

* better representations,
* easier optimization,
* or both.

The exact mechanism remains an important research question.

### What Is the Engineering Complexity?

Industrial adoption depends heavily on implementation simplicity and hardware efficiency.

### What Happens During Inference?

Training efficiency alone is not sufficient.

Inference speed and deployment costs matter enormously in production systems.



## The Deeper Conceptual Insight

The most important idea in the paper may be philosophical rather than architectural.

For years, neural network scaling largely assumed:

> Better models require proportionally larger dense computation.

Virtual Width Networks challenge that assumption.

The paper suggests that:

* representation capacity,
* optimization behavior,
* and compute cost

may be partially separable.

That idea aligns with many recent advances in AI efficiency research.

Modern architectures increasingly try to:

* route computation selectively,
* activate sparse subnetworks,
* externalize memory,
* or compress expensive operations.

VWN extends this trend into the domain of model width.



## Final Takeaway

Virtual Width Networks propose a compelling new idea:

> Neural networks may benefit from being “virtually wider” even when their expensive backbone computations remain relatively compact.

If the approach scales successfully, it could become an important new direction for efficient AI training.

The paper is still early, and many questions remain open.

But its central insight is both elegant and potentially powerful:

> Representation capacity and compute capacity may not need to scale together.

That single idea could influence how future large language models are designed.


## Short Version (TL;DR)

* Wider transformers usually learn better representations.
* But increasing width is computationally expensive.
* Virtual Width Networks attempt to separate representation width from compute width.
* The method creates a larger “virtual” representation space while keeping much of the expensive computation smaller.
* The paper reports substantially faster optimization and improved scaling behavior.
* If the results hold at large scale, VWN could become an important new efficiency technique for training future AI systems.



## References
1. [Virtual Width Networks](https://arxiv.org/abs/2511.11238)

