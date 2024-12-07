---
title: "HF Tools"
date: 2022-05-05T00:17:58+08:00
lastmod: 2022-05-05T00:17:58+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- tech
description: "Study DL"
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
    image: "/img/reading.png" #图片路径例如：posts/tech/123/123.png
    caption: "" #图片底部描述
    alt: ""
    relative: false
---

## Models Loading
The difference between `AutoModel` and `AutoModelForCausalLM` is that the former doesn't have `lm_head`. Thus, `AutoModel` is usually used for embedding model training. `AutoModelForCausalLM` is used for generative application. 


## Model Access
```bash
# Might need to install the pkg
# pip install huggingface_hub
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('MY_HUGGINGFACE_TOKEN_HERE')"



# speedup model download
pip install hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# or in python we can add the following
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" 
```
[1] https://uvadlc-notebooks.readthedocs.io/en/latest/index.html
