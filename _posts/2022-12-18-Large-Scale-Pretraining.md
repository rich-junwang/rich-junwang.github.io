---
layout: post
title: Large Language Model Pretraining
date: 2022-12-18
description: tricks and tips
tags: Research
categories: research
---

Large language model pretraining is a very challenging task which requires very strong engineering and science skills. People tend to underestimate efforts needed to train a good large models like GPT3 etc. Most people imagine that they can get a good language model given enough computation resources. The fact is even today only OpenAI is providing LM APIs where people can freely play with and get good performances. In this blog, we'll talk about pretraining from the whole pipeline: data sourcing, collection and processing, tokenization, architecture engineering and evaluation. Hopefully, it would be helpful for foundational model training practioners. 

### Data
Data is crucial in any ML system. This is true to pretraining as well. As is shown in Gopher paper,  A large, diverse and high-quality dataset is needed to train a good model. In the following table, it shows the datasets used in [`Gopher` model](https://arxiv.org/pdf/2112.11446.pdf) training. Now we're looking at terabytes scale of training data. 

<p align="center">
    <img alt="gopher dataset" src="/assets/img/gopher_data.png" width="80%"/>
    <em>Datasets used in Gopher [2]</em>
    <br>
</p>
An ensuing problem with large amount of data is that data quality is hard to control. In practice, we have to at least make sure the content should be intelligible. 
Diversified datasets are necessary but can't guarantee training success as can be seen from `Gopher` paper, model performs well on QA related tasks but suffers on reasoning task. What else is needed? We'll come back to this later. 

### Tokenizer
Language models compute probability of any string sequence. How to represent the string sequence is determined by tokenizer. Popular options are byte pair encoding (BPE) or wordpiece. As the majority of models are using BPE today, here we focus on BPE based tokenizer. 

As mentioned in GPT2 paper, BPE effectively interpolates between word level inputs for frequent symbol sequences and character level inputs for infrequent symbol sequences. Directly using greedy method to build BPE merging rules can be problematic. For example, word `cat` can be used in a lot of places like `cat?`, `cat!`, `cat.`. One way to solve this issue is to prevent BPE from generating rules across different character categories (letters, digits, puncts etc).

As people are pivoting in-context learing/instruction learning with large models, tokenization efficiency becomes more important. The following tables from Jurassic-1 paper shows the efficiency of tokenizer on several public dataset. 

<p align="center">
    <img alt="tokenization efficiency" src="/assets/img/tokenizer.png" width="100%"/>
    <em>Tokenizer efficiency comparison from [16]</em>
    <br>
</p>

### Model Architecture
All pretrained models are variant of original transformer model. The differences are mainly about it's encoder-decoder architecture or decoder-only architecture. First of all, let's take a look at the choices of available large models. 

| Models &nbsp; &nbsp;  | Model Size &nbsp; &nbsp;   | Token Size &nbsp; |  Architecture  | 
|----|:----:| :----:|
| GPT3 | 175B | 300B | Decoder | 
| OPT | 175B| 300B | Decoder | 
| PaLM | 540B| 300B | Decoder | 
| Gopher | 280B| 300B | Decoder | 
| Chinchilla | 70B| 1400B | Decoder | 
| Jurassic-1 | 178B| - | Decoder | 
| Megatron-Turing NLG | 530B| 270B | Decoder | 
{:.mbtablestyle}




### Training Design
Most of today's pretraining follow suits of a multi-stage and multi-task training. As is shown by Yao in [1], GPT series model is pretrained in such way as well. 
<p align="center">
    <img alt="gopher dataset" src="/assets/img/gpt-lineage.png" width="80%" height=auto/> 
    <br>
    <em>GPT Model Lineage. Image from [1]</em>
    <br>
</p>

From the lineage diagram, we can see that `ChatGPT` model comes from `Codex` model which can be seen as a different stage of training. The way of scheduling tasks and data during training can have great impact on the final model performance. 

#### Critical Batch Size
Research [5] shows that there is a critical batch size in pretraining. When training batch size exceeds critical batch size, model performance starts to degrade. Critical batch size is independent of model size and is related to loss. 

### Evaluation Misconception
A lot of large models come out every year and many claims that they could beat GPT3 model in a wide range of benchmarks like `SuperGlue`, `CLUE` etc. However, when you do benchmark these models in zero-shot setting or some less common tasks (but still very reasonable ones), these models tend to perform really bad. I personally tested `GPT3` model (175b) and `UL2` model (20b) on text2sql and sql2text task, GPT3 gives way better performance. You may argue that the model size differs a lot. However, we can think the other way around: maybe their model training is not easy/efficient to scale to such level. Essentially, what I want to say is that good performance on popular benchmark datasets doesn't mean much for large LM pretraining as this is highly related to source of training data, whether or not doing fine-tuning, proper prompting etc. 


### Stability
During the model training, the most commonly seen issue is gradient exploding, aka, gradient becomes `NaN`. As layers go deeper, this problem happens more often because the way backpropagation works. Over the years, people have proposed many different ways to solve the challenge. 
As is shown in this paper, `On Layer Normalization in the Transformer Architecture`, the post-LN shows stability issue without carefully designed warming-up stage. As a result, they are proposing pre-LN to alleviate the problem. 












## References
[1] [How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources](https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1) <br>
[2] [Gopher: Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/pdf/2112.11446.pdf) <br>
[3] [UL2: Unifying Language Learning Paradigms](https://arxiv.org/pdf/2205.05131.pdf) <br>
[4] [Bloom: Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model](https://arxiv.org/abs/2211.02001) <br>
[5] [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) <br>
[6] [GPT: Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) <br>
[7] [GPT2: Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) <br>
[8] [GPT3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) <br>
[9] [InstructGPT: Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) <br>
[10] [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332) <br>
[11] [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068) <br>
[12] [OPT2: OPT-IML Scaling Language Model Instruction Meta Learning through the Lens of Generalization](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT-IML) <br>
[13] [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) <br>
[14] [Flan-PaLM: Scaling Instruction-Finetuned Language Models](https://arxiv.org/pdf/2210.11416.pdf) <br>
[15] [Chinchilla: Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) <br>
[16] [Jurassic-1: Technical details and evaluation.](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf) <br>
