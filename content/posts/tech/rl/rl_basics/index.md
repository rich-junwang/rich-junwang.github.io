---
title: "RL Basics"
date: 2020-07-18T00:18:23+08:00
lastmod: 2020-07-18T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
    - ML
description: "RL basics"
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

### RL Basics
There are two fundamental problems in the sequential decision making process: reinforcement learning and planning.
In reinforcement learning, the environment is unknown and agent interacts with environment to improve its policy. Within reinforcement learning there are two kinds of operation: prediction and control. Prediction is given policy, evaluate the future. Control is to optimize the future to find the optimal policy. In RL, we alternatively do predition and control to get the best policy. 

In terms of methods, RL algorithm can be categorized into two types: model-free algorithm and model-based algorithm. 
In model-free algorithms, we don't want to or we can't learn the system dynamics. We sample actions and get corresponding rewards to optimize policy or fit a value function. It can further be divied into two methods: policy optimization or value learning.

- Planning
    - Value Iteration: Value iteration uses dynamic programming to compute the value function iteratively using Bellman equation.
    - Policy iteration — Compute the value function and optimize the policy in alternative steps 
- RL
    - Value-learning/Q-learning: Without an explicit policy, we fit the value-function or Q-value function iteratively with observed rewards under actions taken by an off-policy, like an ε-greedy policy which selects action based on the Q-value function and sometimes random actions for exploration.
    - Policy gradients: using neural network to approximate policy and optimize policy using gradient ascent.


### Concepts
A return is a measured value (or a random variable), representing the actual discounted sum of rewards seen following a specific state or state/action pair. Value function is the expected return function. 
The discounted return for a trajectory is defined as:
$$
U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \gamma^3 R_{t+3} + ...
$$

Consequently, the action-value function is defined as
$$
Q_{\pi}(s_t, a_t) = \mathbb{E_t}[U_t|S_t=s_t, A_t=a_t]
$$

State-value function (or value function) can be calculated as:
$$
V_{\pi}(s_t) = \mathbb{E_A}[Q_{\pi}(s_t, A)] = \sum_a \pi(a|s_t) \cdot Q_{\pi}(s_t, a)
$$

#### Bellman Equations
From the definition of the 

### References
1. [RL — Reinforcement Learning Algorithms Overview](https://jonathan-hui.medium.com/rl-reinforcement-learning-algorithms-overview-96a1500ffcda)
2. [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) 

<!-- [x] https://github.com/ShangtongZhang/reinforcement-learning-an-introduction -->
