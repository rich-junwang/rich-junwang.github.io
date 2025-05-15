---
title: "veRL"
date: 2024-08-05T00:18:23-07:00
lastmod: 2024-08-05T00:18:23-07:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- RL
description: "RL training framework"
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

## Hybrid Controller System

RL training can be modeled as a dataflow, where each node represents computation of a neural network (NN) and each edge denotes data dependencies between the NNs. For instance, actor, critic, reward model, reference model, rollout engine each can be seen as a node. The data transfer in-between is the edge. This is an important concept as later when we talk about intra-node computation and inter-node dataflow, the `node` here refers to theses models, not a physical machine we usually talk about in distributed training setting. 

The reason we talk about this is because we have to understand why veRL talks about single-controller and multi-controller system. 
Single-controller, simply put, is one process that coordinates all the workflow as opposed to multi-controller is multiple processes each manages its own workflow. 

Now it becomes obvious that if we use multi-controller system to coordinate dataflow from rollout model to critic, reward model etc, it could be complicated because it's multi-processes communication. Thus, for the dataflow, we want to have single-controller for inter-node data exchange. On the contrary, for modern model training and model inference, people tend to use single-process-multiple-data (SPMD) approach, which is a multi-controller approach. In the ideal case, if both training engine and rollout engine utilize SPMD, parameter update from training engine to rollout engine can be done in parallel from multiple copies which can greatly increase system robustness. vLLM recently added SPMD inference [2].

The overall execution workflow of veRL is shown below:

<div align="center"> 
<img src=images/driver_worker.png style="width: 90%; height: auto;"/> 
</div>


## RL Workflow

A high-level synchronous workflow of RL training process can be summarized as below. There are mainly three steps:
1. Rollout to get trajectory
2. Collect experience, i.e. inference to generate training data
3. Actor/Critic model training


```python
for prompts in dataloader:

    # Stage 1: rollout (response generation): prefill + decoding step by step
    rollout = actor.generate_sequences(prompts)

    # Stage 2: prepare experience: inference
    # i.e. generating training data
    values = critic.compute_values(rollout)
    ref_log_prob = reference.compute_log_prob(rollout)
    reward = reward.compute_reward(rollout, ref_log_prob)
    batch = compute_advantages(reward, values)

    # Stage 3: actor and critic training
    critic_metrics = critic.update_critic(batch)
    actor_metrics = actor.update_actor(batch)

```

### Hybrid Engine

Modern LLM training engine and inference engine gradually diverges. Inferences focuses on leverages customized kernels to minimize latency while training aims to support high precision, large batches and N-D parallelism to maximize throughput. Thus, veRL chooses to have both training engine and rollout/inference engine for actor model. 

Given that we have actor training engine, rollout engine, critic model, reference model and reward model, it's critical to optimize GPU allocation and placement of the models. There are generally three placement plans:
1. Separate placement: each model placed on separate devices
2. Group coloate: different group of models can be placed together
Actor/Ref colocated on some GPUs
Critic/RM colocated on other GPUs
3. All model colocate together




(to be continued)




## References
1. HybridFlow: A Flexible and Efficient RLHF Framework
2. https://github.com/vllm-project/vllm/issues/11400
3. https://verl.readthedocs.io/en/latest/hybrid_flow.html#id9
4. GSPMD: General and Scalable Parallelization for ML Computation Graphs

<!-- 1. https://zhuanlan.zhihu.com/p/30876678559
2. https://zhuanlan.zhihu.com/p/27676081245
3. https://zhuanlan.zhihu.com/p/24682036412 -->

