---
title: "Makefile"
date: 2020-01-11T00:18:23+08:00
lastmod: 2020-01-11T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- Makefile
description: "make sense of makefile"
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
---



### Makefile Syntax

A Makefile consists of a set of *rules*. A rule generally looks like this:

```
targets: prerequisites
    command
    command
    command`
```

* The *targets* are file names, separated by spaces. Typically, there is only one per rule.
* The *commands* are a series of steps typically used to make the target(s). These *need to start with a tab character*, not spaces.
* The *prerequisites* are also file names, separated by spaces. These files need to exist before the commands for the target are run. These are also called *dependencies*




### Commands and execution

#### Command Echoing/Silencing

Add an `@` before a command to stop it from being printed

