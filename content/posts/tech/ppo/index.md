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

assuming that the $x_i$ here is the i.i.d samples from the distribution $p(x)$. So we say that Monte Carlo Approximation is to use one or more samples to calculate the expectation of a distribution. 

#### Importance Sampling
In reality, it could be very challenging to sample data according to the distribution $p(x)$ as it is usually unknown to us. A workaround is to have another known distribution $q(x)$, and define the expectation as:
$$
\mathbb{E_{x\sim p(x)}}[f] = \int{q(x)\frac{p(x)}{q(x)}f(x)} dx
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
&= \mathbb{E_A}\left[  \frac{\partial{\log\pi (a|s; \theta)}}{\partial{\theta}} \cdot Q_{\pi}(s, a) \right]
\end{aligned}
$$
The last step assumes that $\frac{\partial{\log\pi (a|s; \theta)}}{\partial{\theta}} \cdot Q_{\pi}(s, a)$ follows a distribution of $\pi(a|s_t; \theta)$ with respect to the random variable $A$.

Let's take another look at the policy gradient here. First, in practice, when we calculate the expectation we can use Monte Carlo Approximation. The gradient here becomes summations as below:

$$
\nabla_{\theta}(J(\theta)) = \sum_{t} \nabla_{\theta}{\log\pi (a|s; \theta)} \cdot Q_{\pi}(s, a)
$$

This is also called Monte Carlo policy gradient. Since gradient is a direction, this formula shows that policy gradient estimation is the direction of the steepest increase in reward/return. When reward is larger, the policy gradient will be larger.


#### REINFORCE
Since $Q_{\pi}(s, a)$ is the expectation of the return, we can once again use Monte Carlo approximation,
$$
\begin{aligned}
Q_{\pi}(s_t, a_t) &= u_t \\\
&= \sum_{i=t}^{N} {\gamma^{i-t} \cdot r_{i}}
\end{aligned} 
$$
The above MCPG actually gives us a practical algorithm to do policy gradient based RL. Let's summarize it as follows:
1. Play one episode of game to get the trajectory: $s_1, a_1, r_1, s_2, a_2, r_2, ...$
2. Estimate all $q_t \approx u_t$ using above equation
3. Differentiate policy network to get $d_{\theta, t}$
4. Compute policy gradient $g(a_t, \theta_t) = q_t \cdot d_{\theta, t}$


#### Advantage Function and Generalized Advantage Estimation
The above equation is the vanilla policy gradient method. More policy gradient algorithms are proposed later to reduce high variance of the vanilla version. John Schulman's [GAE paper](https://arxiv.org/pdf/1506.02438.pdf) summarized all the improvement methods. In the derivation, the policy gradient is represented as
$$
\frac{\partial{V(s; \theta)}}{\partial{\theta}} = \mathbb{E_A}\left[  \frac{\partial{\log\pi (a|s; \theta)}}{\partial{\theta}} \cdot  \hat{A_t}(s, a) \right]
$$
where $\hat{A_t}(s, a)$ is the advantage function. In implementation, we construct loss function in a way such that the policy gradient $g$ equals to the above result
$$
L(\theta) = \mathbb{E_t}\left[ \log\pi (a|s; \theta) \hat{A_t}(s, a)  \right]
$$

The idea is that the Advantage function calculates how better taking that action at a state is compared to the average value of the state. It’s subtracting the mean value of the state from the state action pair. Mathematically, $A(s_t, a_t) = Q(s_t, a_t) − V (s_t)$, where $Q(s_t, a_t)$ is the action-value function, representing the expected return after taking action at at state $s$, and $V (s_t)$ is the value function, representing the average expected return at state st.

#### Actor-Critic Algorithm
There we give a recap of how actor-critic method works. In Actor-Critic algorithm, we use one neural network $\pi(a|s; \theta)$ to approximate policy function $\pi(a|s)$ and use another neural network $q(s, a; w)$ to approximate value function $Q_{\pi}(s, a)$.
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
Essentially, the algorithm alternates between sampling and optimization. The expectation in the above equation indicates that we need to average over a finite batch of empirical samples. 


#### Proximal Policy Optimization
Vanilla policy gradient method uses on-policy update. Concretely, the algorithm samples empirical data from a policy network $\pi_{\theta}$ parameterized with $\theta$. After updating the network itself, the new policy network is $\pi_{\theta_{new}}$ and the old policy $\pi_{\theta}$ is out of use and future sampling will be from $\pi_{\theta_{new}}$. This whole process is not efficient enough. The solution to this is to reuse the old samples to achieve off-policy training. From above importance sampling section, we know that:

$$
\mathbb{E_{x\sim p(x)}}\left[f \right] = \mathbb{E_{x\sim q(x)}} \left[ \frac{p(x_i)}{q(x_i)}f(x_i) \right]
$$

Similarly, we can make a change to the objective function of our policy gradient, and the resulting policy gradient will become
$$
\begin{aligned}
g &= \mathbb{E_{{(s_t, a_t)} \sim \pi_{\theta}}}\left[  \frac{\partial{\log\pi (a_t|s_t; \theta)}}{\partial{\theta}} \cdot  \hat{A_t}(s, a) \right] \\\
&= \mathbb{E_{{(s_t, a_t)} \sim \pi_{\theta_{old}}}}\left[ \frac{\pi_{\theta}(a_t|s_t; \theta)}{\pi_{\theta_{old}}(a_t|s_t)} \frac{\partial{\log\pi (a_t|s_t; \theta)}}{\partial{\theta}} \cdot  \hat{A_t}(s, a) \right]
\end{aligned}
$$
Consequently, the loss becomes

$$
L(\theta) = \mathbb{E_{{(s_t, a_t)} \sim \pi_{\theta_{old}}}}\left[ \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A_t}(s, a)  \right]
$$
This is so-called surrogate objective function. In the above section, we mentioned how to use chain rule to get the expectation format of gradient, here we just to reverse the process to get the above loss function. 


In the importance sampling section, we saw that the variance of new distribution could be large when the proposal distribution is not so close to the original distribution. Thus, to deal with this, people add KL diveragence to the loss function to limit the old and new policy difference. Using Largrangian dual method, we can add this constraint to the objective function:

$$
L(\theta) = \mathbb{E_{{(s_t, a_t)} \sim \pi_{\theta_{old}}}}\left[ \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A_t}(s, a)   - \beta KL[\pi_{\theta_{old}}(a_t|s_t), \pi_{\theta}(a_t|s_t)]\right]
$$


### Implementation
For language generation task, generating a token is an action. Agent is the target language model we want to train.

Here we first look at the implementation from Deepspeed-chat model. The actor-critic algorithm requires to load four model in training: actor model, critic model, reference model and reward mdoel. Actor model is the poliy network and critice model is the value network. Reference model and reward model are frozen in training. Reference model is used to contrain the actor model predictions so that they won't divege too much. Reward model gives the current step reward.




### References
[1] [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438.pdf) <br>
[2] [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf) <br>
[3] [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) <br>
[4] [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) <br>
[5] https://cs.uwaterloo.ca/~ppoupart/teaching/cs885-spring18/schedule.html <br>
[6] https://github.com/wangshusen/DRL <br> 
[7] https://www.davidsilver.uk/teaching/ <br>
[8] [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/pdf/1909.08593.pdf) <br>
[9] https://zhuanlan.zhihu.com/p/677607581 <br>
[10] [DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales
](https://arxiv.org/abs/2308.01320) <br>
[11] [Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/pdf/2307.04964.pdf) <br>
[12] [Secrets of RLHF in Large Language Models Part II: Reward Modeling](https://arxiv.org/pdf/2401.06080.pdf) <br>
[13] [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/abs/2403.17031) <br>
[14] [Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study](https://arxiv.org/pdf/2404.10719.pdf) <br>
[15] [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/abs/2005.12729) <br>
[16] [Advanced Tricks for Training Large Language Models with Proximal Policy Optimization](https://difficult-link-dd7.notion.site/eb7b2d1891f44b3a84e7396d19d39e6f?v=01bcb084210149488d730064cbabc99f)
<!-- [11] [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) -->
<!-- [12] [PPO for beginners](https://github.com/ericyangyu/PPO-for-Beginners) -->
<!-- [13] [A Survey of Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2312.14925) -->