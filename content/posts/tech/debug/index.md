---
title: "Efficient Debugging"
date: 2023-06-18T00:18:23+08:00
lastmod: 2023-07-18T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
    - Debug
description: "How to efficiently solve practical problem"
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
    image: "" #图片路径例如：posts/tech/123/123.png
    caption: "" #图片底部描述
    alt: ""
    relative: false
math: true
---



## Attach GDB
```
/usr/local/cuda/bin/cuda-gdb attach process-id
bt

gdb python process-id
```

## Python Debug
```Python
# python code
import pdb; pdb.set_trace()
```


## Tools

### Shortcuts
These shortcuts can be redefined in the keymap(Pycharm) or Keyboard shortcuts (VScode), but we need to know the meaning of these keys. 
Pycharm
- reformatting: CMD + ALT + L -> convert json line format to json format
- join lines: CTRL + SHIFT + J -> convert json format to json line format. When in use, we can select the parts we want to join.

VSCode
- reformatting: SHIFT + ALT + F -> convert json line format to json format
- join lines: CTRL + J -> convert json format to json line format