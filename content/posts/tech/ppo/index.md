---
title: "PPO and Its Implementation"
date: 2023-7-05T00:18:23+08:00
lastmod: 2023-07-05T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- blog
description: "Proximal Policy Optimization"
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
In this blog, I'll go through the theory (simplified version) of PPO algorithm and try to code it from scratch.
### Basics

#### Monte Carlo Approximation
Distributions of random variables in reality are mostly unknown. Sampling-based methods are extensively used in practice becaue of its ease of use and the generality where they can be applied. One of the fundamental problems is to calculate the expectation of a random variable, which can be expressed as 

$$
\mathbb{E_{x\sim p(x)}}\left(f(x)\right) = \int{f(x)p(x)} dx
$$
when it's a continuous random variable with a probability density function of $p$, or 
$$
\mathbb{E}\left(f(x)\right) = \sum_x{f(x)p(x)}
$$
when it's a discrete random variable with probability mass function of $p$.
Then the Monte Carlo approximation says that the expectation is:
$$
\mathbb{E}\left(f(x)\right) \approx \frac{1}{N}\sum_{i=1}^{N}{f(x_i)}
$$

assuming that the $x_i$ here is the i.i.d samples from the distribution $p(x)$.

#### Importance Sampling
In reality, it could be very challenging to sample data according to the distribution $p(x)$ as it is usually unknown to us. A workaround is to have another known distribution $q(x)$, and define the expectation as:
$$
\mathbb{E}[f] = \int{q(x)\frac{p(x)}{q(x)}f(x)} dx
$$
This can be seen as the expectation of function $\frac{p(x)}{q(x)}f(x)$ according to the distribution of $q(x)$. The distribution is sometimes called the **proposal distribution**. Then the expectation can be estimated as
$$
\mathbb{E_{x\sim q(x)}}[f] \approx \frac{1}{N}\sum_{i=1}^{N}{\frac{p(x_i)}{q(x_i)}f(x_i)}
$$
Here the ratios $\frac{p(x_i)}{q(x_i)}$ are referred sa the importance weights.
The above derivation looks nice. However, we need to notice that the although the expectation is similar in both cases, the variance is different:

$$
Var_{x\sim p(x)}[f] = \mathbb{E_{x\sim p(x)}}[f(x)^2] - ({\mathbb{E_{x\sim p(x)}}[f(x)]})^2
$$

$$
\begin{aligned}
Var_{x\sim q(x)}[f] &= \mathbb{E_{x\sim q(x)}}[({\frac{p(x_i)}{q(x_i)}f(x_i)})^2] - (\mathbb{E_{x\sim q(x)}}[{\frac{p(x_i)}{q(x_i)}f(x_i)}])^2 \\\
&= \mathbb{E_{x\sim p(x)}}[{\frac{p(x_i)}{q(x_i)}f(x_i)^2}] - (\mathbb{E_{x\sim p(x)}}[f(x_i)])^2
\end{aligned}
$$
Notice that the second equation here, in the second step derivation, the expectation is relative to distribution of $p(x)$. From the above two equations, we can see that to make the sampling distribution as close as possible to the original distribution, the ratio $\frac{p(x_i)}{q(x_i)}$ has to be close to 1.

#### Policy Gradient
First, let's remind ourselves some basics. The discounted return for a trajectory is defined as:
$$
U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \gamma^3 R_{t+3} + ...
$$

Consequently, the action-value function is defined as
$$
Q_{\pi}(s_t, a_t) = \mathbb{E_t}[U_t|S_t=s_t, A_t=a_t]
$$

State-value function can be calculated as:
$$
V_{\pi}(s_t) = \mathbb{E_A}[Q_{\pi}(s_t, A)] = \sum_a \pi(a|s_t) \cdot Q_{\pi}(s_t, a)
$$

In policy gradient algorithm, the policy function $\pi(a|s_t)$ is approximated by policy network $\pi(a|s_t; \theta)$. $\theta$ here is the neural network model parameters. Then the policy-based learning is to maximize the objective function 
$$
\begin{aligned}
J(\theta) &= \mathbb{E_S}[V(S; \theta)] \\\
&=  \sum_{s\in S} d_{\pi}(s) V_{\pi}(s_t; \theta) \\\
&= \sum_{s\in S} d_{\pi}(s) \sum_a \pi(a|s_t; \theta) \cdot Q_{\pi}(s_t, a)
\end{aligned}
$$

where $d_{\pi}(s)$ is the stationary distribution of Markov chain for $\pi_{\theta}$, namely the state distribution under policy $\pi$.
Now we know the objective function of the policy-based algorithm, we can learn the parameters $\theta$ through policy gradiet ascent. 

Now we can look at how to get the policy gradient. Since the first summation of the last step in the above equation has nothing to do with $\theta$, so we can focus on getting the derivatives of the value function $V_{\pi}(s; \theta)$. Using chain rule, it's easy to get:
$$
\begin{aligned}
\frac{\partial{V(s; \theta)}}{\partial{\theta}} &= \sum_a \frac{\partial{\pi (a|s; \theta)}}{\partial{\theta}} \cdot Q_{\pi}(s, a) \\\
&= \sum_a \pi(a|s_t; \theta) \frac{\partial{\log\pi (a|s; \theta)}}{\partial{\theta}} \cdot Q_{\pi}(s, a) \\\
&= \mathbb{E_A}\left[  \frac{\partial{\log\pi (A|s; \theta)}}{\partial{\theta}} \cdot Q_{\pi}(s, A) \right]
\end{aligned}
$$
The last step assumes that $\frac{\partial{\log\pi (A|s; \theta)}}{\partial{\theta}} \cdot Q_{\pi}(s, A)$ follows a distribution of $\pi(a|s_t; \theta)$ with respect to the random variable $A$.

The above equation is the vanilla policy gradient method. More policy gradient algorithms are proposed later to reduce high variance of the vanilla version. John Schulman's [GAE paper](https://arxiv.org/pdf/1506.02438.pdf) summarized all the improvement methods. 

#### Actor-Critic Algorithm
There we give a recap of how actor-critic method works. In Actor-Critic algorithm, we use one neural network $\pi(a|s; \theta)$ to approximate $\pi(a|s)$ and use another neural network $q(s, a; w)$ to approximate $Q_{\pi}(s, a)$
- Observe state $s_t$, and randomly sample action from policy  $a_t \sim \pi(\cdot | s_t; \Theta_t)$
- Let agent perform action $a_t$, and get new state $s_{t+1}$ and reward $r_t$ from environment
- Randomly sample $\tilde{a}_{t+1} \sim \pi(\cdot | s_t; \Theta_t)$ without performing the action
- Evaluate value network: $q_t = q(s_t, a_t; W_t)$ and $q_{t+1} = q(s_{t+1}, \tilde{a}_{t+1}; W_t)$
- Compute TD error: $\delta_t = q_t - (r_t + \gamma \cdot q_{t+1})$
- Differentiate value network: $d_{w,t} = \frac{\partial{q(s_t, a_t, w)}}{\partial{w}}$ (autograd will do this for us)
- Update value network: $ w_{t+1} = w_t -  \alpha \cdot \delta_t  \cdot d_{w, t}$
- Differentiate policy network: $ d_{\theta, t} = \frac{\partial{\log\pi (a|s; \theta)}}{\partial{\theta}} $ (again autograd will do this for us)
- Update policy network: $\theta_{t+1} = \theta_t + \beta \cdot q_t \cdot d_{\theta, t}$.
    - We can also use: $\theta_{t+1} = \theta_t + \beta \cdot \delta_t \cdot d_{\theta, t}$ to update policy network. This is called policy gradient with baseline.
