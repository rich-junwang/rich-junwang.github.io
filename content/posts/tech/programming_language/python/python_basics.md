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
- Python
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

## Concurrency in Python
Parallelism consists of performing multiple operations at the same time. Multiprocessing is a means to effect parallelism, and it entails spreading tasks over a computer’s central processing units (CPUs, or cores). Multiprocessing is well-suited for CPU-bound tasks: tightly bound for loops and mathematical computations usually fall into this category.

Concurrency is a slightly broader term than parallelism. It suggests that multiple tasks have the ability to run in an overlapping manner. (There’s a saying that concurrency does not imply parallelism.)

Threading is a concurrent execution model whereby multiple threads take turns executing tasks. One process can contain multiple threads. Python has a complicated relationship with threading thanks to its GIL, but that’s beyond the scope of this article.

What’s important to know about threading is that it’s better for IO-bound tasks. While a CPU-bound task is characterized by the computer’s cores continually working hard from start to finish, an IO-bound job is dominated by a lot of waiting on input/output to complete.

To recap the above, concurrency encompasses both multiprocessing (ideal for CPU-bound tasks) and threading (suited for IO-bound tasks). Multiprocessing is a form of parallelism, with parallelism being a specific type (subset) of concurrency. The Python standard library has offered longstanding support for both of these through its multiprocessing, threading, and concurrent.futures packages.


### Shared Memory

**Why We Use Shared Memory?**

When using Python for parallel processing, performance may not always scale as expected—especially due to how data is shared between tasks. Here's why:

- Threading

Python threads share the same memory space, making data sharing straightforward. However, due to the Global Interpreter Lock (GIL), threads are best suited for I/O-bound operations—not CPU-intensive workloads.

- Multiprocessing

In contrast, the multiprocessing module spawns separate processes, each with its own memory space. Sharing data between processes requires serialization via Queue, Pipe, or similar mechanisms. This copying and (de)serialization introduce significant communication overhead, which can negate performance gains.

- A Better Approach: Shared Memory

To reduce inter-process communication overhead, shared memory allows multiple processes to access the same memory block directly. This eliminates the need for serialization and copying, enabling much faster data exchange and improved scalability for CPU-bound tasks. If you're hitting performance bottlenecks with multiprocessing, consider using multiprocessing.shared_memory for efficient, zero-copy data sharing.

```python
from multiprocessing import Process, shared_memory
import numpy as np


def worker(shm_name, size):
    """Process that modifies shared memory"""
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    array = np.ndarray((size,), dtype=np.float64, buffer=existing_shm.buf)

    array += 10  # Modify data in shared memory
    existing_shm.close()


if __name__ == "__main__":
    size = 5
    shm = shared_memory.SharedMemory(create=True, size=size * np.float64().nbytes)
    array = np.ndarray((size,), dtype=np.float64, buffer=shm.buf)
    array[:] = [1, 2, 3, 4, 5]  # Initialize array

    print("Before:", array)

    # Create a process that modifies shared memory
    p = Process(target=worker, args=(shm.name, size))
    p.start()
    p.join()

    print("After:", array)  # Values modified by worker process

    # Cleanup
    shm.close()
    shm.unlink()
```

Note that when aloocating memory, the unit is byte. The allocation such as shown below would create 1MB memory. 
```python
shm = SharedMemory(name="myshm", create=True, size=2**20) 
```


## update python on ubuntu
When there are multiple version of python in the system, how to set the default python to use. Below we suppose to install newer version of python3.9
```bash
sudo apt install python3.9

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.[old-version] 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2

#after the following command, select the new version when being prompted and press enter
sudo update-alternatives --config python3

# There is a simpler way. From system admin perspective, this is not scalable.
sudo ln -sf /usr/bin/python3.9 /usr/bin/python3
```

## find a python package related files
```
pip show -f package-name

pip list | grep kafka
```

### Find out python binry file path
```python
>>> import streamlit
>>> print(streamlit.__file__)
```


## Process Got Killed
Once in a while, I found my python process got killed without any errors. Most of time, it's related to out of memory (OOM) issue. We can quickly check that using the following command
```
dmesg | grep "oom-kill" | less
```


## Virtual Env
```bash
python3 -m pip install --upgrade pip
python3 -m pip install --user virtualenv
python3 -m venv .venv
source .venv/bin/activate

# exit
deactivate
```

## Debug
```python
# using pdb
# use `n` to step over
# use `s` to step into
# use `ll` to check current lines
# see more here: https://realpython.com/python-debugging-pdb/
import pdb; pdb.set_trace()


# using ipdb
if torch.distributed.get_rank() == 0:
    import ipdb; ipdb.set_trace()


# using ipdb. Program will enter ipython at the exception
from ipdb import launch_ipdb_on_exception
def filter_even(nums):
    for i in range(len(nums)):
        if nums[i] % 2 == 0:
            del nums[i]

with launch_ipdb_on_exception():
    print(filter_even(list(range(6))))
# we can check internal stack, `i`, `nums` etc and can also execute the next step. 
    
# We can also use the following way. The program will error out and enter ipdb.
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(call_pdb=1)

def filter_even(nums):
    for i in range(len(nums)):
        if nums[i] % 2 == 0:
            del nums[i]

filter_even(list(range(6)))

```

## Exception
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
# or
traceback.print_exc()
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

## new method
* The __new__() is a static method of the object class.
* When you create a new object by calling the class, Python calls the __new__() method to create the object first and then calls the __init__() method to initialize the object’s attributes.
* Override the __new__() method if you want to tweak the object at creation time.


## Hacky Way to Add File into PYTHONPATH
```python
curr_file_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(curr_file_path)
sys.path.append(os.path.dirname(curr_file_path))
```

## Subprocess
```python
import subprocess
# Download
dl = subprocess.Popen(["git", "clone", str(repo_path), str(repo_dir)])

# It also accepts str as the input command
output_path = "my_out"
cmd = """python3 train.py --local-rank 0"""
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=open(f"{output_path}.stderr", "wt", encoding="utf-8"),
    shell=True,
    encoding='utf-8',
    bufsize=0)

```


## Dataclass
Data classes use something called a default_factory to handle mutable default values. To use default_factory, we need to use the field() specifier.
```python
from dataclasses import dataclass, field
from typing import List

RANKS = '2 3 4 5 6 7 8 9 10 J Q K A'.split()
SUITS = '♣ ♢ ♡ ♠'.split()

def make_french_deck():
    return [PlayingCard(r, s) for s in SUITS for r in RANKS]

@dataclass
class Deck:
    # we can't do the following way to assign a mutable default value to a field. 
    # cards: List[PlayingCard] = make_french_deck()

    cards: List[PlayingCard] = field(default_factory=make_french_deck)

```
A few commonly used parameters that field supports
- default: Default value of the field
- default_factory: Function that returns the initial value of the field
- init: Use field in .__init__() method? (Default is True.)
- repr: Use field in repr of the object? (Default is True.)



## Global Variable in Python
Python looks for variables in four different scopes:

> 1. The local, or function-level, scope, which exists inside functions
> 2. The enclosing, or non-local, scope, which appears in nested functions
> 3. The global scope, which exists at the module level
> 4. The built-in scope, which is a special scope for Python’s built-in names

The code snippet below shows how these scopes work. 
```python
# Global scope

def outer_func():
    # Non-local scope
    def inner_func():
        # Local scope
        print(some_variable)
    inner_func()
```
Notice that global is only at module level. There is no program level global variable for python. After we define a global variable in one module, we could use it in other module with its module name.  




## UV

Right now, my workflow with UV is to maintain a shared virtual environment. That way, I don’t have to reinstall packages for every single project, and I can still use the Python interpreter to run code directly on my local setup. For some tools, I lean on uv tool to handle the installation. I only spin up dedicated virtual environments when a project’s dependencies look like they could get messy.

```bash

# show all pythons available
uv python list

# fix python for uv
uv python pin 3.12

# create shared env
uv venv ~/.uv/shared --python 3.12
source ~/.uv/shared/bin/activate

alias shared='source ~/.uv/shared/bin/activate' 
alias dev='source ~/.uv/dev/bin/activate' 
alias local='source .venv/bin/activate'

# install through pip after running shared command (aliased above) 
uv pip install pandas

# install common tools with uv tool
uv tool install mytool
```


### Other Common Usages
```bash

# install
pip install uv

# Add the following line to zshrc or bashrc after pip install to get the binary in PATH
# export PATH=`python3 -m site --user-base`/bin:\$PATH

# initialize a project
uv init

# create virtual env
uv venv
uv venv .my_venv_path

# specify python version in venv
uv venv -p 3.11
uv venv -python 3.11

source .my_venv_path/bin/activate


# install packages using uv
uv pip install pandas 

# update package
uv pip install -U pandas

# install from local directory
uv pip install .
# install from local, support editable
uv pip install -e .


# uninstall 
uv pip uninstall pandas

# uv get package versions
uv pip compile my_packages.txt -o requirements.txt
uv pip compile - -o requirements.txt
uv pip sync requirements.txt


uv cache prune
uv cache clean
```



## Python Development Mode

```bash
pip install -e path/to/SomeProject
```

Editable installs allow you to install your project without copying any files. Instead, the files in the development directory are added to Python’s import path. This approach is well suited for development and is also known as a “development installation”.



## References
1. https://realpython.com/python-data-classes/
2. https://realpython.com/python-debugging-pdb/
