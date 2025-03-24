---
title: "Network for Cluster"
date: 2024-09-08T12:01:14-07:00
lastmod: 2024-09-08T12:01:14-07:00
author: ["Jun"]
keywords: 
- 
categories: # 没有分类界面可以不填写
- 
tags: # 标签
- 
description: "Network.."
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



### Test Network Bandwidth
First step to verify instance networking. For instance, AWS P4 is supposed to have 400G networking. 

```bash
# on server node 1
iperf3 -s


# on server node 2
iperf3 -c node_1_ip


# for high performance compute cluster we can do
# -O 3, ignore first 3 seconds
# -P 32 use 32 parallel streams
# -t 30, run for 30 seconds
iperf3 -c node_1_ip -P 32 -O 3 -t 30

```