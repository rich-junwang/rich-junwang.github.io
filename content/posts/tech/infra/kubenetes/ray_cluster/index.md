---
title: "Ray Cluster"
date: 2024-08-08T12:01:14-07:00
lastmod: 2024-08-08T12:01:14-07:00
author: ["Jun"]
keywords: 
- 
categories: # 没有分类界面可以不填写
- 
tags: # 标签
- 
description: "Ray Cluster.."
weight:
slug: ""
draft: true # 是否为草稿
comments: true # 本页面是否显示评论
reward: false # 打赏
mermaid: true #是否开启mermaid
showToc: true # 显示目录
TocOpen: true # 自动展开目录
hidemeta: false # 是否隐藏文章的元信息，如发布日期、作者等
disableShare: true # 底部不显示分享栏
showbreadcrumbs: true #顶部显示路径
cover:
    image: "images/speedup.jpg" #图片路径例如：posts/tech/123/123.png
    zoom: # 图片大小，例如填写 50% 表示原图像的一半大小
    caption: "" #图片底部描述
    alt: ""
    relative: false
---



## Basics
Today let's talk about using kuberay operator to build ray cluster for large scale reinforcement learning training. 

We have to configure three main roles of a ray cluster, which are the head pod, the worker pods, and the shared file system that will be attached to both the head and worker pods.



## References
1. kuberay: https://github.com/ray-project/kuberay
2. https://kubernetes.io/docs/reference/kubectl/cheatsheet/
3. https://catalog.workshops.aws/sagemaker-hyperpod-eks/en-US
4. https://catalog.workshops.aws/sagemaker-hyperpod/en-US
5. https://github.com/aws-samples/aws-do-ray
