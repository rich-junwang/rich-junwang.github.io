---
layout: post
title: InstructGPT and ChatGPT
date: 2022-11-25
description: tricks and tips
tags: Research
categories: research
---
Recently ChatGPT model has demonstrated remarkable success of large pretrained language model being able to generate coherent, logical and meaningful conversations. While as of this writing, the corresponding paper is still not available yet. In this blog, I'll dive deep into InstructGPT model to see what's under the hood of this model. 
### Issues with Traditional LM
Language modeling objective is trying to predict next token given all previous tokens. However, when we're prompting LM in inference time, we hope LM can generate things based on our instructions/intent instead of merely predicting the most likely tokens. This is the so-called `misalignment` between training and inference. 

### Solution
Using reinforcement learning to learn human feedback. For this purpose, they have to collect a dataset. The way to collect the dataset is as follows: 
- select some contract labeler
- collect human written prompt-answer pairs. Prompts are either from GPT3 API or from human annotation.
- collect a dataset of human-labeled comparisons between outputs from our models on a larger set of API prompts.

The following diagram from the paper demonstrated how these steps unfold during the training. 
<p align="center">
    <img alt="make it parse" src="/assets/img/instructgpt.png" width="800"/>
    <br>
</p>

In summary, there are three steps: 
- Use labeled data to fine-tune GPT3 model
- Train a reward model
- Use RL to optimize GPT3 parameters

In the first step, we got data from annotators and use this data to fine-tune GPT3 model. In the second step, they prepare some questions and GPT3 model gives multiple predictions for each question and annotators are asked to rank the generated predictions. This data is used to train reward model. The reward model is used for prediction and predict which one is most close to human choice. Reward model gives a score and the closer to human choice, the higher of the score. 

Finally, use policy-based RM algorithm to do further optimization. The whole process is shown in the diagram below. It uses reward mechanism to train model. The reward can be seen as the loss function in traditional ML training. Reward function is much more versatile than loss function (Think about DotaRL and AlphaGo). The consequence is that reward function may not be differentiable, thus can't be used for back-propagation. People can sample rewards to proximate this loss function.

<p align="center">
    <img alt="rl" src="/assets/img/lm_rl.png" width="800"/>
    <br>
    <em>RL algorithm. Image from [1]</em>
    <br>
</p>

(To be continued)




### References
[1] [Learning to summarize from human feedback](https://arxiv.org/pdf/2009.01325.pdf) <br>
[2] [InstructGPT: Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) <br>
[3] [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/pdf/1909.08593.pdf) <br>
[4] https://github.com/lvwerra/trl  <br>
[5] https://zhuanlan.zhihu.com/p/590311003  <br>
[6] [Super-natural instructions: generalization via declarative instructions on 1600+ NLP tasks](https://arxiv.org/abs/2204.07705)
