---
title: "Distributed Training Infra"
date: 2022-05-05T00:18:23+08:00
lastmod: 2022-05-05T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- blog
description: "Distributed training infrastructure"
weight:
slug: ""
draft: false # 是否为草稿
comments: true
reward: true # 打赏
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

Distributed infrastructure is a big and interesting topic. I don't work on infrastructure side, but I run into the concepts a lot, so I create this blog to help me understand more about infrastructure.

Most of today's distributed framework involves three parts, collective communication, data loading and preprocessing and distributed scheduler. We'll look into these three parts resepectively.


## Collective Communication
We can start with point to point communication. Normally point to point communication refers to two processes communication and it's one to one communication. Accordingly, collective communication refers to 1 to many or many to many communication. In distributed system, there are large amount of communications among the nodes. 

There are some common communication ops, such as Broadcast, Reduce, Allreduce, Scatter, Gather, Allgather etc.  

### Broadcast and Scatter
Broadcast is to distribute data from one node to other nodes. Scatter is to distribute a portion of data to different nodes. 
<p align="center">
    <img alt="flat sharp minimum" src="images/broadcast_and_scatter.png" width="40%" height=auto/> 
    <br>
    <em>MPI broadcast and scatter</em>
    <br>
</p>



### Reduce and Allreduce
Reduce is a collections of ops. Specifically, the operator will process an array from each process and get reduced number of elements.
<!-- ![](images/reduce1.png) -->
<p align="center">
    <img alt="flat sharp minimum" src="images/reduce1.png" width="60%" height=auto/> 
    <br>
    <em>MPI reduce</em>
    <br>
</p>

<p align="center">
    <img alt="flat sharp minimum" src="images/reduce2.png" width="60%" height=auto/> 
    <br>
    <em>MPI reduce</em>
    <br>
</p>

Allreduce means that the reduce operation will be conducted throughout all nodes.
<p align="center">
    <img alt="flat sharp minimum" src="images/allreduce.png" width="60%" height=auto/> 
    <br>
    <em>MPI Allreduce</em>
    <br>
</p>
