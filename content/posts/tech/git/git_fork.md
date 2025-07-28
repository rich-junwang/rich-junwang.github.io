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

## Three-way Code Management
![alt text](./image.png)
```bash
# Assuming the current directory is <your-repo-name>
git remote add upstream https://github.com/alshedivat/al-folio.git
git fetch upstream

# check all branches
git branch -avv

git checkout main # local main tracking personal repo
git rebase upstream/main # rebase from upstream
git push -f # push to personal repo
```



## Manually Add Upstream Branch 
We can also manually do the changes. This is the original .git/config

```
[core]
    repositoryformatversion = 0
    filemode = true
    bare = false
    logallrefupdates = true
[remote "origin"]
    url = https://github.com/yyy/zzz
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
    url = https://github.com/xxx/zzz
    fetch = +refs/heads/*:refs/remotes/origin/*
[remote "upstream"]
    url = https://github.com/yyy/zzz
    fetch = +refs/heads/*:refs/remotes/upstream/*
[branch "master"]
    remote = origin
    merge = refs/heads/master
```


## Sync New Commits from Public Repo


The best approach is:
```bash
git fetch upstream # this make sure we don't merge

# rebase if you want to rebase
git rebase upstream/main

# merge if you want to merge
git merge upstream/main

# It's not recommended to do git pull as the following. Because git pull will do two things
# - fetched and merged upstream/<branch> into your current branch
# - creates a merge commit
git pull upstream

```

If we haven’t make changes at master, we can do:

```bash
git checkout master
git pull upstream master  # pull public (upstream) to local branch
git push origin master
```

If there is conflict with your changes:

```bash
git checkout master
git fetch upstream
git merge --no-ff upstream/master
git push origin master
```



Or simply 
```bash
git checkout main
git rebase upstream/main
```
