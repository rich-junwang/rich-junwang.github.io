---
title: "Using Git to Create a Private Fork"
date: 2018-02-11T00:18:23+08:00
lastmod: 2018-02-11T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- Tool
description: "Create a fork"
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
---

1. make changes in .git/config

This is the original .git/config

```
[core]
    repositoryformatversion = 0
    filemode = true
    bare = false
    logallrefupdates = true
[remote "origin"]
    url = https://github.com/ElementAI/duorat
    fetch = +refs/heads/*:refs/remotes/origin/*
[branch "master"]
    remote = origin
    merge = refs/heads/master
```


Making changes like this: 

```
[core]
    repositoryformatversion = 0
    filemode = true
    bare = false
    logallrefupdates = true
[remote "origin"]
    url = https://github.com/rich-junwang/duorat
    fetch = +refs/heads/*:refs/remotes/origin/*
[remote "upstream"]
    url = https://github.com/ElementAI/duorat
    fetch = +refs/heads/*:refs/remotes/upstream/*
[branch "master"]
    remote = origin
    merge = refs/heads/master
```


I guess we can also do this by the following command:

```
git remote add upstream https://github.com/ElementAI/duorat
```


2. Push your current mainline to my private remote

```
git push origin master
```

## Sync New Commits from Public Repo

If we haven’t make changes at master, we can do:

```
git checkout master
git pull upstream master  # pull public (upstream) to local branch
git push origin master
```

If there is conflict with your changes:

```
git checkout master
git fetch upstream

git merge --no-ff upstream/master

git push origin master
```



```
# Assuming the current directory is <your-repo-name>
$ git remote add upstream https://github.com/alshedivat/al-folio.git
$ git fetch upstream
```

