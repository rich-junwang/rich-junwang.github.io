---
# title: "The Rise of the Omni-Modal AI"
title: "VLM"
date: 2024-05-18T00:18:23+08:00
lastmod: 2024-05-18T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
    - Multimodality
description: "Multimodality"
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

## ViT

ViT (Vision Transformer) flattens the input image into a 2D sequence of patches (e.g., 16x16), and then uses a linear projection layer to convert each patch into a fixed-length feature vector—similar to word embeddings in natural language processing. Additionally, each patch is assigned a positional index, which is mapped to a positional vector through a positional embedding layer. The patch embeddings and positional embeddings are summed together to form the input to the Transformer encoder, resembling the design of the BERT model.
<div align="center"> <img src=images/vit.png style="width: 70%; height: auto;"/> Figure 1. ViT </div>

ViT has a BERT encoder based architecture and it uses a learnable [CLS] token to obtain a representation of the entire image, which is then passed through a fully connected layer to serve downstream classification tasks. When pretrained on large-scale datasets and transferred to medium- or small-scale image recognition benchmarks (e.g., ImageNet, CIFAR-100, VTAB), ViT achieves better performance than CNN-based models, while significantly reducing computational resource requirements during training.

The key to ViT’s success lies in large-scale pretraining. Without this step, directly training a ViT model on standard open datasets would still result in performance inferior to CNNs, which benefit from stronger inductive biases.

Training datasets are supervised datasets including: ImageNet 1.3M images, ImageNet-21k, 21k classes and 14 images, JFT dataset, 18k classes and 303M images. The `[CLS]` token is used for classification task at training time. 

## CLIP

The core idea of CLIP is to align visual and natural language representations through contrastive learning. CLIP first extracts features separately from text and images. The text encoder is a pretrained BERT, while the image encoder can be a traditional CNN model or a ViT (Vision Transformer) model. After obtaining the image and text embeddings, their features are normalized, and cosine similarity is calculated between all image-text pairs in the batch. A contrastive loss function—such as triple Loss or InfoNCE Loss is used to pull together positive pairs and push apart negative pairs.
<div align="center"> <img src=images/clip.png style="width: 100%; height: auto;"/> Figure 2. CLIP </div>

After pretraining on a large number of image-text pairs, CLIP learns a shared representation space for both the text and image encoders. There are typically two types of downstream applications:

Fine-tuning the model on downstream tasks to adapt it for specific image-text matching tasks. In this case, the text or image encoder can also be used for unimodal tasks. Zero-shot learning, where the pretrained image-text representations are directly used without any further training. Zero-shot usage of CLIP is shown in diagrams (2) and (3). For an image classification task, each candidate class is converted into a textual prompt using a template like “A photo of a {object}”, where object is the candidate class. For a given image to be classified, CLIP extracts a visual embedding via the image encoder and compares it to the embeddings of all candidate text prompts obtained from the text encoder. The class with the highest similarity is selected as the prediction.

Due to its simple architecture and outstanding performance, CLIP has been widely adopted in many subsequent works, with its pretrained backbone often used as an initialization for visual representation modules.

## BLIP-2
The core idea of BLIP-2 is to enhance multimodal performance and reduce training costs by leveraging pretrained vision and language models. The architecture consists of a vision encoder, a vision-language adapter (Q-Former), and a large language model (LLM) layer. 
<div align="center"> <img src=images/blip2.png style="width: 65%; height: auto;"/> Figure 3. Blip-2 Architecture </div>

- Vision Encoder: Uses a ViT model with weights initialized through CLIP pretraining. The final enhancement layer (which enriches output features) is removed. During training, the weights are frozen and not updated.
- LLM: Early versions of BLIP-2 used OPT/FlanT5 to experiment with decoder-based and encoder-decoder-based LLMs. This part is also frozen during training and not updated. 
- Q-Former: The Q-Former is a module to compress the visual feature sequence and to extract key visual information via learned query vectors and feeds it to the LLM. The size of learnable queries is like 32 x 768 in the paper. This component holds most of the trainable parameters during multimodal model training.

<div align="center"> <img src=images/blip2_qformer.png style="width: 100%; height: auto;"/> Figure 4. Blip-2 Q-former </div>

## VLM

### Qwen-VL

Qwen-VL is a classic large multimodal models. Qwen-VL uses the Qwen-7B LLM as the language backbone, OpenCLIP-pretrained ViT-bigG as the visual feature encoder, and a randomly initialized single-layer cross-attention module as the adapter between vision and language. The total number of parameters is approximately 9.6 billion.

As shown in the figure, Qwen-VL’s training process is divided into three stages:
<div align="center"> <img src=images/qwen1.png style="width: 90%; height: auto;"/> Figure 5. Qwen-vl </div>

- stage 1: Pretraining – The goal is to align the visual encoder and the LLM using large-scale image-text pairs. During this stage, the LLM's parameters are frozen, only training adaptor and ViT. Training data is 1.4B weakly labeled, web-crawled set of image-text pairs. The training objective is to minimize the cross-entropy of the text tokens.
- stage 2: Multitask Pretraining – Uses higher-quality, multitask image-text data (mainly from open-source vision-language datasets and some in-house data), higher image resolution, and involves full model fine-tuning.
- stage 3: Instruction Tuning – The visual encoder is frozen; training data is mostly generated via self-instruction from the LLM. This phase aims to improve instruction following and multi-turn dialogue capabilities.

Another important insight from Qwen-VL is that stages 2 and 3 incorporate not only vision-language data but also pure-text data. This helps preserve the LLM’s capabilities and avoid catastrophic forgetting. This strategy has proven effective in other models as well. Compared to models such as InstructBLIP, Qwen-VL simplifies the adapter between vision and language, using only a shallow attention pooling module nad achieving better performance.


### LLaVA

Below shows an overview of LLaVA 1.5’s data and architecture. Compared to Qwen-VL, LLaVA 1.5 uses much less pretraining and instruction tuning data (with both Stage 2 and Stage 3 of Qwen-VL viewed as instruction tuning for comparison). In terms of architecture, both the vision encoder and the LLM use different base models, and the adapter between vision and language is an even simpler MLP layer. On several evaluation benchmarks, LLaVA 1.5 outperforms Qwen-VL, suggesting that with the right optimizations, it's possible to achieve strong multimodal understanding with less data and simpler architecture. As shown in the data comparison, improvements can come from:

<div align="center"> <img src=images/llava1.5.png style="width: 90%; height: auto;"/> Figure 6. LlaVA-1.5 </div>

- Adding high-quality, fine-grained VL data
- Enriching instruction templates
- Using pure-text instruction tuning
- Increasing image resolution
- Scaling up the LLM parameter size


### Qwen2-VL

Qwen2-VL represents a contemporary architecture of VLM. The model architecture is shown below. It features several innovations comparing with previous version:
1. Naive Dynamic Resolution mechanism: The model adapts to varying image resolutions by generating different numbers of visual tokens on-the-fly—moving away from traditional fixed-resolution processing. Qwen2-VL pretrained their own ViT model based on this mechanism. 
2. Multimodal Rotary Position Embedding: Introduces rotary-style positional embeddings capable of jointly encoding positional information across text, images, and video. This unified encoding supports seamless fusion of positional cues across modalities
3. Unified Image & Video Processing Pipeline: The Visual Transformer (ViT) backbone (675M parameters) is structured to process both static images and video frames in a single, coherent paradigm. Simplifies multimodal modeling and demonstrates strong performance on both image and video tasks .

Notice that there is no explicit vision adaptor outside the vision module. 
<div align="center"> <img src=images/qwen2.png style="width: 100%; height: auto;"/> Figure 7. Qwen2-VL </div>

Training comprises three stages:
- First stage: ViT pretraining. Note that the training only optimizes ViT module, but LLM is used in the training process which differs from the original ViT pretraining. 
- Second stage: unfreeze all parameters and train with a wider range of data for more comprehensive learning.
- Third stage: lock the ViT parameters and perform exclusive fine-tuning of the LLM using instructional datasets


#### Multimodal Rotary Position Embedding

Extending RoPE from 1D to 2D: For a given position, split the input vector of dimension $d$ into two halves. Use a 1D RoPE matrix of size $\frac{d}{2}$ (denoted as $R_x$) to process the first half of the vector, and use another 1D RoPE matrix of size $\frac{d}{2}$ (denoted as $R_y$) to process the second half. Then, concatenate the two processed halves together — this completes the 2D RoPE processing.

**3D-RoPE** mapping is similar to 2D-RoPE. For a given position $(x, y, z)$, split the input vector of dimension $d$ into three equal parts. Use a 1D RoPE matrix for $x$ (denoted as $\mathcal{R}_x$) to process the first part of the vector, use a 1D RoPE matrix for $y$ (denoted as $\mathcal{R}_y$) to process the second part, and use a 1D RoPE matrix for $z$ (denoted as $\mathcal{R}_z$) to process the third part. Finally, concatenate the three processed parts together.






## LWM


LWM's core goal is to create **multimodal large model capable of ultra-long context understanding**. With support for 1 million token inputs, LWM can comprehend videos over an hour long. The key highlights of LWM’s work include:

* Support for ultra-long contexts, capable of handling extended text, image sequences, or video content;
* Solutions for key technical challenges: Mixed-length input using a *Masked Sequence Packing* method; balancing visual and textual modalities via *loss weighting*; automatic generation of long-sequence Q\&A datasets for model training;
* Implementation of high-performance optimizations such as *RingAttention* and *Masked Sequence Packing*, enabling training on multimodal sequences of million-token scale;
* Open-sourcing of a 7B-parameter large model, including long-context text-only models (LWM-Text, LWM-Text-Chat) and multimodal models (LWM, LWM-Chat).

LWM uses a Transformer-based structure and extends the context length limit on top of LLaMA 2 (7B). LWM uses VQGAN as the visual encoder, which encodes 256 × 256 input images into 16 × 16 discrete tokens. This enables LWM not only to generate text, but also to generate image tokens from text, which can be reconstructed into video. For multiple images or video frames, visual features can be extracted individually and input into the LLM along with textual modalities.

The model training process consists of two main stages:

- Extend the context length of the text-only LLM to 1M using the Books dataset;
- Multimodal training with long context, using mixed image-text data, video-text data, and pure-text Books data.

There are two core challenges across these stages:

1. Scalable training of long documents;
2. How to stably extend the LLM’s context length.

The first focuses on training efficiency and computational cost, while the second emphasizes the effectiveness of long-context extension. To address challenge 1, LWM implements efficient RingAttention, combined with FlashAttention. For challenge 2, on one hand, both training stages adopt multi-round training to gradually increase context length. On the other hand, the model’s positional encoding capability for long texts is enhanced by simply adjusting the \$\theta\$ parameter in RoPE.





## References
1. World Model on Million-Length Video And Language With RingAttention
2. CLIP: Learning Transferable Visual Models From Natural Language Supervision
3. Improved Baselines with Visual Instruction Tuning
4. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
5. [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966)
6. [Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution](https://arxiv.org/pdf/2409.12191)
7. [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
8. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
9. World Model on Million-Length Video And Language With Blockwise RingAttention


<!-- 5. https://github.com/wdndev/mllm_interview_note -->
<!-- https://zhuanlan.zhihu.com/p/25267823390 -->
<!-- https://zhuanlan.zhihu.com/p/7352653203 -->
<!-- https://zhuanlan.zhihu.com/p/28205969434 -->