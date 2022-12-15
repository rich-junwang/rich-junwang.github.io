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
    <img alt="make it parse" src="/assets/instructgpt.png" width="800"/>
    <br>
<p>


(To be continued)
