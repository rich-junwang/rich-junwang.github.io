---
title: "Loss in ML"
date: 2019-06-18T00:18:23+08:00
lastmod: 2019-07-18T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
    - ML
description: "Losses in ML"
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

### Sigmoid
Sigmoid is one of the most used activation functions. 
$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$
It has nice mathematical proprities:
$$
\sigma^\prime(x) = \sigma(x) \left[ 1 - \sigma(x) \right]
$$
and
$$
\left[log\sigma(x)\right]^\prime = 1 - \sigma(x) \\\
\left[log\left(1 - \sigma(x)\right)\right]^\prime = - \sigma(x)
$$

### Logistic Regression
For a binary classification problem, for an example $x = (x_1, x_2, \dotsb , x_n)^T$, the hypothesis function (for the positive class) can be written as:
$$
\begin{aligned}
h_{\theta}(1|x) &= \sigma(\theta_0 + \theta_1x_1 + \theta_2x_2 + \dotsb + \theta_nx_n) \\\
&= \sigma(\theta^{\mathrm{T}}x) \\\
&= \frac{1}{1+e^{-\theta^{\mathrm{T}}x}}
\end{aligned}
$$
Consequently, for the negative class,
$$
\begin{aligned}
h_{\theta}(0|x) &= 1 - \frac{1}{1+e^{-\theta^{\mathrm{T}}x}} \\\
&= \frac{1}{1+e^{\theta^{\mathrm{T}}x}} \\\
&= \sigma(-\theta^{\mathrm{T}}x)
\end{aligned}
$$

Single sample cost function of logistic regression is expressed as:
$$
L(\theta) = -y_i \cdot \log(h_\theta(x_i)) -  (1-y_i) \cdot \log(1 - h_\theta(x_i))
$$
Notice that in the second term $1 - h_\theta(x_i)$ is the negative class probability

<!-- For my understanding, for each data example, there is only one label. So we never really sum positive label and negative label loss together for one example. We only sum them together for two training examples, say one has positive label and one has negative label.  -->


### Cross Entropy
Cross entropy defines the distance between model output distribution and the groudtruth distribution.
$$
H(y,p) = -\sum_{i}y_i \log(p_i)
$$
Since the $y_i$ is the class label (1 for positive class, 0 for negative), essentially here we're summing up the negative log probably of the positive label. What is the reason why we say that negative log likehood and cross entropy is equivalent. 

When normalization function (we can say activation function of last layer) is softmax function, namely, for each class $s_i$ the probability is given by
$$
f(s)_i = \frac{e^{s_i}}{ \sum_j ^{C} e^{s_j}}
$$
Given the above cross entropy equation, and there is only one positive class, the softmax cross entropy loss is:
$$
L = -log(\frac{e^{s_p}}{ \sum_j ^{C} e^{s_j}})
$$
here $p$ stands for positive class. 
If we want to get the gradient of loss with respect to the logits ($s_i$ here), for positive class we can have
$$
\frac{\partial{L}}{\partial{s_p}} = \left(\frac{e^{s_p}}{ \sum_j ^{C} e^{s_j}} - 1 \right) \\\
\frac{\partial{L}}{\partial{s_n}} = \left(\frac{e^{s_n}}{ \sum_j ^{C} e^{s_j}} - 1 \right)
$$
We can put this in one equation, which is what we commonly see as the graident of cross entropy loss for softmax
$$
\frac{\partial{L}}{\partial{s_i}} = p_i - y_i
$$
$p_i$ is the probability and $y_i$ is the label, 1 for positive class and 0 for negative class.

### Binary Cross Entropy
In the above section, we talked about softmax cross entropy loss, here we talk about binary cross entropy loss which is also called Sigmoid cross entropy loss. 

We apply sigmoid function to the output logits before BCE. Notice that here we apply the function to each element in the tensor, all the elements are not related to each other, this is why BCE is widely used for **multi-label** classification task. 

For each label, we can calculate the loss in the same way as the logistic regression loss. 

```python
import torch
import numpy as np

pred = np.array([[-0.4089, -1.2471, 0.5907],
                [-0.4897, -0.8267, -0.7349],
                [0.5241, -0.1246, -0.4751]])

# after sigmod, pred becomes
# [[0.3992, 0.2232, 0.6435],
# [0.3800, 0,3044, 0.3241],
# [0.6281, 0.4689, 0.3834]]

label = np.array([[0, 1, 1],
                  [0, 0, 1],
                  [1, 0, 1]])

# after cross entropy, pred becomes
# [[-0.5095, -1.4997, -0.4408], take neg and avg 0.8167
# [-0.4780, -0.3630, -1.1267], take neg and avg 0.6559
# [-0.4651, -0.6328, -0.9587]] take neg and avg 0.6855
# 0 * ln0.3992 + (1-0) * ln(1-0.3992) = -0.5095
# (0.8167 + 0.6559 + 0.6855) / 3. = 0.7194

pred = torch.from_numpy(pred).float()
label = torch.from_numpy(label).float()

crition1 = torch.nn.BCEWithLogitsLoss()
loss1 = crition1(pred, label)
print(loss1) # 0.7193

crition2 = torch.nn.MultiLabelSoftMarginLoss()
loss2 = crition2(pred, label)
print(loss2) # 0.7193
```




### Noise Contrastive Estimation
Noise contrastive estimation or negative sampling is a commonly used computation trick in ML to deal with expansive softmax computation or intractable partition function in computation.

The derivation of NCE loss sometimes can be bewildering, but the idea is actually very simple. For example, in word2vec implementation, the negative sampling is to choose 1 positive target and 5 negative target, and calculate the binary cross entropy loss (binary logistic loss) and then do backward propagation. 



### Contrastive Loss


### CLIP Loss
CLIP loss is the same with the paper from [3]. The negatives here are used for contrastive learning. However, they're not using NCE method like word2vec. It's more like softmax cross entropy.

```python
import torch
from torch import nn
import torch.nn.functional as F

import config as CFG
from modules import ImageEncoder, TextEncoder, ProjectionHead


class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    CLIP = CLIPModel()
    loss = CLIP(batch)
    print("")
```



### Reference
1. [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020.pdf)
2. http://www.cnblogs.com/peghoty/p/3857839.html
3. [contrastive learning of medical visual representations from paired images and text](https://arxiv.org/abs/2010.00747)
4. https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/CLIP.py
