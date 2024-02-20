---
title: "Python"
date: 2015-04-09T12:01:14-07:00
lastmod: 2015-04-09T12:01:14-07:00
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
reward: true # 打赏
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

### Concurrency in Python
Parallelism consists of performing multiple operations at the same time. Multiprocessing is a means to effect parallelism, and it entails spreading tasks over a computer’s central processing units (CPUs, or cores). Multiprocessing is well-suited for CPU-bound tasks: tightly bound for loops and mathematical computations usually fall into this category.

Concurrency is a slightly broader term than parallelism. It suggests that multiple tasks have the ability to run in an overlapping manner. (There’s a saying that concurrency does not imply parallelism.)

Threading is a concurrent execution model whereby multiple threads take turns executing tasks. One process can contain multiple threads. Python has a complicated relationship with threading thanks to its GIL, but that’s beyond the scope of this article.

What’s important to know about threading is that it’s better for IO-bound tasks. While a CPU-bound task is characterized by the computer’s cores continually working hard from start to finish, an IO-bound job is dominated by a lot of waiting on input/output to complete.

To recap the above, concurrency encompasses both multiprocessing (ideal for CPU-bound tasks) and threading (suited for IO-bound tasks). Multiprocessing is a form of parallelism, with parallelism being a specific type (subset) of concurrency. The Python standard library has offered longstanding support for both of these through its multiprocessing, threading, and concurrent.futures packages.


### update python on ubuntu
When there are multiple version of python in the system, how to set the default python to use. Below we suppose to install newer version of python3.9
```
sudo apt install python3.9

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.[old-version] 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2

#after the following command, select the new version when being prompted and press enter
sudo update-alternatives --config python3

# There is a simpler way. From system admin perspective, this is not scalable.
sudo ln -sf /usr/bin/python3.9 /usr/bin/python3
```

### find a python package related files
```
pip show -f package-name
```

### Find out python binry file path
```python
>>> import streamlit
>>> print(streamlit.__file__)
```


### Process Got Killed
Once in a while, I found my python process got killed without any errors. Most of time, it's related to out of memory (OOM) issue. We can quickly check that using the following command
```
dmesg | grep "oom-kill" | less
```


### Virtual Env
python3 -m pip install --upgrade pip
python3 -m pip install --user virtualenv
python3 -m venv .venv
source .venv/bin/activate


### Debug
```
# using pdb
import pdb; pdb.set_trace()
```

### Exception
How to catch generic exception type
```python
try:
    someFunction()
except Exception as ex:
    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    print(message)
```

The difference between the above and using just `except` without any argument is twofold: (1) A bare except doesn't give you the exception object to inspect (2) The exceptions SystemExit, KeyboardInterrupt and GeneratorExit aren't caught by the above code, which is generally what you want.

If you also want the same stacktrace you get if you do not catch the exception, you can get that like this (still inside the except clause):
```python
import traceback
print(traceback.format_exc())
```

If you use the logging module, you can print the exception to the log (along with a message) like this:
```python
import logging
log = logging.getLogger()
log.exception("Message for you, sir!")
```

To dig deeper and examine the stack, look at variables etc., use the post_mortem function of the pdb module inside the except block:
```python
import pdb
pdb.post_mortem()
```