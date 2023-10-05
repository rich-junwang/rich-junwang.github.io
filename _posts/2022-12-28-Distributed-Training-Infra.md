---
layout: post
title: Distributed Training Infrastructure
date: 2022-12-26
description: tricks and tips
tags: Software
categories: Software
---

Distributed infrastructure is a big and interesting topic. I don't work on infrastructure side, but I run into the concepts a lot, so I create this blog to help me understand more about infrastructure.

Most of today's distributed framework involves three parts, collective communication, data loading and preprocessing and distributed scheduler. We'll look into these three parts resepectively.


## Collective Communication
We can start with point to point communication. Normally point to point communication refers to two processes communication and it's one to one communication. Accordingly, collective communication refers to 1 to many or many to many communication. In distributed system, there are large amount of communications among the nodes. 

There are some common communication ops, such as Broadcast, Reduce, Allreduce, Scatter, Gather, Allgather etc.  

### Broadcast and Scatter
Broadcast is to distribute data from one node to other nodes. Scatter is to distribute a portion of data to different nodes. 
<p align="center">
    <img alt="flat sharp minimum" src="/assets/img/distributed_infra/broadcast_and_scatter.png" width="60%" height=auto/> 
    <br>
    <em>MPI broadcast and scatter</em>
    <br>
</p>



### Reduce and Allreduce
Reduce is a collections of ops. Specifically, the operator will process an array from each process and get reduced number of elements. 
<p align="center">
    <img alt="flat sharp minimum" src="/assets/img/distributed_infra/reduce1.png" width="60%" height=auto/> 
    <br>
    <em>MPI reduce</em>
    <br>
</p>

<p align="center">
    <img alt="flat sharp minimum" src="/assets/img/distributed_infra/reduce2.png" width="60%" height=auto/> 
    <br>
    <em>MPI reduce</em>
    <br>
</p>

Allreduce means that the reduce operation will be conducted throughout all nodes.
<p align="center">
    <img alt="flat sharp minimum" src="/assets/img/distributed_infra/allreduce.png" width="60%" height=auto/> 
    <br>
    <em>MPI Allreduce</em>
    <br>
</p>
