---
title: "Scaling LLMs in Depth"
date: 2025-03-05T00:18:23+08:00
lastmod: 2025-03-05T00:18:23+08:00
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

Is pretraining hitting a wall? While, history tells us that it's always good to scale. Today we're studying the challenges in scaling transformer models in depth dimension. 

## Challenges

Standard transformers suffer from:

- exploding or vanishing gradients,
- exploding activations,
- rank collapse (token representations becoming too similar),
- instability from large attention logits (QK values).



## Scale LLMs in Depth

When building Large Language Models (LLMs), the modern machine learning playbook usually screams one word: Wider! To make models more capable, we scale up their hidden dimensions and add billions of parameters. But why don't we see networks that are incredibly deep—say, hundreds or thousands of layers?

The answer lies in a hidden chaos that plagues deep architectures: signal instability. As data travels forward and gradients flow backward through hundreds of layers, the network effectively "blinds" or "deafens" itself, resulting in exploding gradients, vanishing signals, or token representations collapsing into redundant mush.

Recently I found this paper [1] has cracked this depth barrier. By replacing guesswork with elegant mathematics, the researchers successfully trained architectures up to 1,000 layers deep.



### The Architecture Problem: Linear Growth vs. Exponential Decay

To understand why deep networks break, we have to look at the plumbing of a classic Transformer block. Transformers rely on *residual connections* (or skip-connections), where the input to a layer is added directly to its output ($x_{out} = x_{in} + \text{Block}(x_{in})$).

Depending on where you place your Layer Normalization (LayerNorm), two bad things happen when you scale depth ($N$):

1. Pre-LN (The modern standard)

Because every layer's output is stacked directly onto the main highway, the variance of the forward signal explodes linearly ($\mathcal{O}(N)$). When backpropagating, the gradients don't play nice either—they grow hyperbolically, causing extreme training instability.


2. Post-LN

If the normalization happens before the highway merge, the backward gradient variance vanishes exponentially ($\mathcal{O}(c^{\pm N})$). The lower layers learn absolutely nothing.



Faced with this, engineers usually resort to "profiling passes"—running empirical tests to find magic scaling fractions to dampen the layers. But these hacks degrade performance and reduce the network's expressivity.



### The Solution: A Unified Signal Propagation Theory

Instead of guessing, the authors did something beautifully old-school: they calculated the exact closed-form mathematical equations for how signal moments (mean and variance) propagate through every single micro-component of a Transformer. They mapped everything from the Embeddings to Softmax, Attention, and Dropout.

Crucially, unlike previous theories, they didn't assume data was completely random or independent. They accounted for the fact that real text has highly correlated words (like "the" or "and" popping up repeatedly).

When tested against real language and vision datasets, their mathematical formulas predicted the actual behavior of the models with an astonishing 99.8% accuracy ($R^2 = 0.998$)—even up to 768 layers deep! 



### Introducing DeepScaleLM: Bending the Curve to $\mathcal{O}(1)$

Armed with this bulletproof mathematical map, the researchers designed DeepScaleLM (DSLM).

DSLM introduces a precise initialization and fixed residual scaling scheme that acts like an adaptive shock absorber for the network. It scales down the weight initializations and adjusts the residual highway so that both the forward signal variance and the backward gradient variance remain perfectly conserved at $\mathcal{O}(1)$. No matter if the model has 12 layers or 1,000 layers, the signal variance stays completely stable.



### Why Depth Beats Width

You might wonder: Is a 1,000-layer model actually useful, or is it just a flex? As it turns out, deep and thin beats shallow and wide. In machine learning, depth corresponds to compositionality—the ability of a network to build complex, hierarchical abstractions over multiple steps of reasoning.

The paper's empirical results across Language Modeling, Speech Translation, and Vision Transformers (ViT) were striking:

* A 192-layer model easily outperformed standard 12-layer and 24-layer baselines.


* Even more impressive, a deep DSLM model with fewer parameters consistently beat wider, shallow models that had more than twice the parameter count.


* DeepScaleLM required minimal compute overhead (within 15% wall-clock time of original architectures) but translated to significant gains in downstream Question Answering tasks and downstream model robustness.





The future of LLM design isn't just about throwing more parameters at a model; it's about structuring them elegantly. By unlocking the depth dimension, we open the door to highly efficient, deeply analytical models that can process the complexities of our world—one layer at a time.




### References
1. [Transformers Get Stable: An End-to-End Signal Propagation Theory for Language Models](https://arxiv.org/abs/2403.09635)

