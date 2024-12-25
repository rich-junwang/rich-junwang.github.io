---
title: "Python AsyncIO"
date: 2024-08-09T12:01:14-07:00
lastmod: 2024-08-09T12:01:14-07:00
author: ["Jun"]
keywords: 
- 
categories: # 没有分类界面可以不填写
- 
tags: # 标签
- 
description: "Python Master"
weight:
slug: ""
draft: false # 是否为草稿
comments: true # 本页面是否显示评论
reward: false # 打赏
mermaid: true #是否开启mermaid
showToc: true # 显示目录
TocOpen: true # 自动展开目录
hidemeta: false # 是否隐藏文章的元信息，如发布日期、作者等
disableShare: true # 底部不显示分享栏
showbreadcrumbs: true #顶部显示路径
cover:
    image: "" #图片路径例如：posts/tech/123/123.png
    zoom: # 图片大小，例如填写 50% 表示原图像的一半大小
    caption: "" #图片底部描述
    alt: ""
    relative: false
---

### Introduction
AysncIO in python has two keywords: async/await. Many people who first encounter async concept might wonder isn't that python can only has one thread in execution given the constraint of GIL? 

Indeed, ayncio is bound by GIL and it can't run more than one task at any moment as is shown below. This means that if another thread needs to run, ownership of the GIL must be passed from the current executing thread to the other thread. This is what called preemptive concurrency. This kind of switching is expensive when there are lots of threads. 

The core concept in asyncio is coroutine. asyncio has its own concurrency synchronization through coroutine. It coordinates task switch with little cost. Simply put, python emulate concurrency in one thread through coroutine using event loop. 
<p align="center">
    <img alt="Thread and coroutine" src="images/image.png" width="80%"/>
    <em>Thread and coroutine</em>
    <br>
</p>

Coroutine style synchronization still has its overhead, why we would bother switching tasks? The reason is behind the io part in asyncio. Think about that you have the following three tasks:
>- Task1: cooking rice takes 40 mins
>- Task2: washing clothes takes 30 mins
>- Task3: dish washing takes 30 mins

How much it would take a person to complete all these tasks. It won't take us 100 mins for all these tasks because we just need to kick things off and have machines done for us. On the contrary, the following tasks most likely will consume us 100 mins because we have to get involved attentively. 
>- Task1: watching tv 30 mins
>- Task2: jogging 30 mins
>- Task3: playing video games 40 mins

This example is just how illustrate where async ops help in Python -- only in IO-bound programs such as http requests, file I/O etc, but not in CPU-bound programs. Note that in reality, python won't allow us to coordinate the execution of each tasks. We can only pack tasks and send them for async execution. 

```python
import asyncio
import time


async def async_task():
    now = time.time()
    await asyncio.sleep(1)
    print("Doing async tasks")
    await asyncio.sleep(1)
    print(time.time() - now)


def sync_task():
    now = time.time()
    time.sleep(1)
    print("Doing async tasks")
    time.sleep(1)
    print(time.time() - now)


async def main():
    await asyncio.gather(*[async_task() for _ in range(3)])

now = time.time()
# run 3 async_task() coroutine concurrently
asyncio.run(main())
print(f"Time elapsed for running 3 coroutine tasks: {time.time() - now}")


now = time.time()
# run 3 sync_task() coroutine concurrently
sync_task()
sync_task()
sync_task()
print(f"Time elapsed for running 3 sync tasks: {time.time() - now}")
```

