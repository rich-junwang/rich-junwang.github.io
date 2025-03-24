---
title: "LoRA Fine-tuning"
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

This works because during training, the smaller weight matrices (A and B in the diagram below) are separate. But once training is complete, the weights can actually be merged into a new weight matrix that is identical.

<p align="center">
    <img alt="lora merge" src="images/merge.png" width="80%" height=auto/> 
    <em>LoRA Merge Weights</em>
</p>

### How To Merge LoRA Weights
Here we talk about Deepspeed + Lightning model merge.

1. Merge Deepspeed model into single shard
```python
module_spec = importlib.util.spec_from_file_location("zero_to_fp32", os.path.join(ckpt_path_dir, "zero_to_fp32.py"))
zero_to_fp32 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zero_to_fp32)
zero_to_fp32.convert_zero_checkpoint_to_fp32_state_dict(ckpt_path_dir, single_shard_output_ckpt_path)
``` 

2. Load Base model
Load base model (the model before lora fine-tuning)
```python
base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype="auto")
``` 

3. Get peft model
```python
peft_config = LoraConfig(**json.load(open(args.peft_config)))
# This model has both base and adapter model definition, weights are not initialized
peft_model = get_peft_model(base_model, peft_config)s
```

4. Load LoRA fine-tuned ckpt
Load LoRA fine-tuned checkpoint and replace lightning prefix (`_forward_module`) with peft model prefix (`base_model`). Fix other parameter issues (like model head etc), then put the state_dict into the peft_model from above step.

5. Merge
```python
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(model_save_path)
```

