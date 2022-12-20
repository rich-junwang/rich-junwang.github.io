---
layout: post
title: Large Language Model Pretraining
date: 2022-12-18
description: tricks and tips
tags: Research
categories: research
---

Large language model pretraining is a very challenging task which requires very strong engineering and science skills. People tend to underestimate efforts needed to train a good large models like GPT3 etc. Most people imagine that they can get a good language model given enough computation resources. The fact is even today only OpenAI is providing LM APIs where people can freely play with and get good performances. 

### Data
Data is crucial in any ML system. This is true to pretraining as well. As is shown in Gopher paper,  A large, diverse and high-quality dataset is needed to train a good model. In the following table, it shows the datasets used in [`Gopher` model](https://arxiv.org/pdf/2112.11446.pdf) training. 

<p align="center">
    <img alt="gopher dataset" src="/assets/img/gopher_data.png" width="100%"/>
    <br>
<p>

However, diversified datasets are necessary but can't guarantee training success as can be seen from `Gopher` paper, model performs well on QA related tasks but suffers on reasoning task. What else is needed?

### Training Design
Most of today's pretraining follow suits of a multi-stage and multi-task training. As is shown by Yao in [1], GPT series model is pretrained in such way as well. 


### Evaluation Misconception
A lot of large models come out every year and many claims that they could beat GPT3 model in a wide range of benchmarks like `SuperGlue`, `CLUE` etc. However, when you do benchmark these models in zero-shot setting or some less common tasks (but still very reasonable ones), these models tend to perform really bad. I personally tested `GPT3` model (175b) and `UL2` model (20b) on text2sql and sql2text task, GPT3 gives way better performance. You may argue that the model size differs a lot. However, we can think the other way around: maybe their model training is not easy/efficient to scale to such level. Essentially, what I want to say is that good performance on popular benchmark datasets doesn't mean much for large LM pretraining as this is highly related to source of training data, whether or not doing fine-tuning, proper prompting etc. 


### Stability
During the model training, the most commonly seen issue is gradient exploding, aka, gradient becomes `NaN`. As layers go deeper, this problem happens more often because the way backpropagation works. Over the years, people have proposed many different ways to solve the challenge. 
As is shown in this paper, `On Layer Normalization in the Transformer Architecture`, the post-LN shows stability issue without carefully designed warming-up stage. As a result, they are proposing pre-LN to alleviate the problem. 












## References
[1] [How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources](https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1)
[2] 
