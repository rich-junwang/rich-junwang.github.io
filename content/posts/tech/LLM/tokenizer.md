---
title: "Tokenizer in LLM"
date: 2023-03-05T00:18:23+08:00
lastmod: 2023-03-05T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- LLM
description: "Tokenizer in LLM"
weight:
slug: ""
draft: true # 是否为草稿
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
---

When we build LLMs, the very first step is to build a good tokenizer. Currently, there are two main frameworks to build tokenizers. The first is huggingface tokenizers library which can be used to build tokenizer like GPT series models. The second is sentencepiece from google which is used to build LLaMA series (LLaMA, mistral, Yi-34B etc) tokenizers.

### Byte Level BPE
There are mainly 5 components in the tokenizer.
- Normalizer
- Pretokenizer
- BPE Model
    + WordPiece
    + Unigram
    + Sentencepiece
- PostProcessor
- Decoder

I have been always in favor of sentencepiece style tokenizer because it provides a way to split based on spaces. However there is a catch.
Very recently, I found that Mistral tokenizer is not a lossless tokenizer!!! What does that mean? 
```python
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
s = "good good"
print(tok.tokenize(s))
print(tok.encode(s))
print(tok.decode(tok.encode(s)))
```