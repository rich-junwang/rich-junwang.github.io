---
title: "Git"
date: 2018-02-11T00:18:23+08:00
lastmod: 2018-02-11T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- blog
description: "Some of my frequently used commands"
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


### Git 

#### Git merge
Suppose we're on **master** branch, if we want to override the changes in the master branch with feature branch, we can use the following command
```
git merge -X theirs feature
```

to keep the master branch changes:
```
git merge -X ours feature
```

If we want to rebase of current branch onto the master, and want to keep feature branch
```
git rebase master -X theirs
```

if we want to keep master branch changes over our feature branch, the
```
git rebase master -X ours
```

To summarize, we can have the following table:

<table>
<thead>
<tr>
<th>&nbsp; Currently on</th>
<th>Command &nbsp; &nbsp;</th>
<th>Strategy &nbsp; &nbsp;</th>
<th>Outcome</th>
</tr>
</thead>
<tbody>
<tr>
<td>&nbsp;master</td>
<td>git merge feature &nbsp;</td>
<td>
<strong>-Xtheirs</strong> &nbsp; &nbsp;</td>
<td>Keep changes from feature branch</td>
</tr>
<tr>
<td>master</td>
<td>git merge feature</td>
<td><strong>-Xours</strong></td>
<td>keep changes from master branch</td>
</tr>
<tr>
<td>&nbsp;feature</td>
<td>git rebase master &nbsp;</td>
<td>
<strong>-Xtheirs</strong> &nbsp; &nbsp;</td>
<td>Keep changes from feature branch</td>
</tr>
<tr>
<td>feature</td>
<td>git rebase master</td>
<td><strong>-Xours</strong></td>
<td>keep changes from master branch</td>
</tr>
</tbody>
</table>
{:.mbtablestyle}

<br>
#### Git Diff

To check two branch difference, suppose we're on branch1, then we can do,
```
git diff HEAD..master
```


#### Delete a remote branch

Delete a remote branch
```
git push origin -d remote_branch_name 
```


#### Git rebase
To fixup, squash, edit, drop, reword and many other operations on the previous N commit:
```
git rebase -i HAED~N
```

#### Git commit
```
git commit --amend --no-edit
```

#### Undo git add
The simplest way to undo a git add is to use git reset. It removes staged file, but will keeop the local changes there. 
```
git reset file_path
```


#### Git check difference
Use the following command to checkout **COMMIT** (commit hash) ancestor and COMMIT difference
```
git diff COMMIT~ COMMIT
git diff HEAD~ HEAD
```


#### Git rebase resolve conflicts
Sometimes when we do git rebase and we have many commits, we have to resolve a lot of conflicts, which can be really frustrating. One quick way might be to squash commits first to have a single commit, then do rebase. Another way is what we describe below.


First, checkout temp branch from feature branch and start a standard merge
```
git checkout -b temp
git merge origin/master
git commit -m "Merge branch 'origin/master' into 'temp'"
```

You will have to resolve conflicts, but only once and only real ones. Then stage all files and finish merge.
Then return to your feature branch and start rebase, but with automatically resolving any conflicts.

```
git checkout feature
git rebase origin/master -X theirs
```

Branch has been rebased, but project is probably in invalid state. We just need to restore project state, so it will be exact as on branch 'temp'. Technically we just need to copy its tree (folder state) via low-level command git commit-tree. Plus merging into current branch just created commit.

```
git merge --ff $(git commit-tree temp^{tree} -m "Fix after rebase" -p HEAD)
git branch -D temp
```

More details is here: https://github.com/capslocky/git-rebase-via-merge. Thanks to the original author!


### How to Split Large PR
Sometimes, we have a giant PR which we want to merge. Often times, it gives reviewer a lot of headache. Now we learn how to split large PR into smaller ones. Suppose our feature branch is `my_feature_branch`, then we can get the diff file using:
#### Step 1
```shell
git diff master my_feature_branch > ../huge_pr_file
```
#### Step 2
We switch back to master and create a new feature branch for the first small pr.
```shell
git checkout master
git checkout -b first_small_feature_pr
```

#### Step 3
Whilst on the first small feature branch, we apply all the code changes.
```shell
git apply ../huge_pr_file
```
After running this command, we'll see all the unstaged changes on the `first_small_feature_pr` branch. Now we can stage any files we want, commit and push them. 


#### Step 4
After pushing/committing the first feature pr, we can stash all the remaining changes (so that we won't commit in this branch).
```shell
git stash --include-untracked --keep-index
```

#### Step 5
Repeat this process from above to create a second PR. Based on the dependency or references, we might have to create a new branch based on the other small PR branch.
```shell
git checkout master
git checkout -b second_small_feature_pr

#OR

git checkout first_small_feature_pr
git checkout -b second_small_feature_pr
```



### How to show difference of commits without checkout to the branch
```bash
# If we know the git commit
git diff xxxx~ xxxx

# if we know the branch name
git diff my_branch~ my_branch
```


### Git submodule
```bash
# Add submodule
git submodule add -b branch_name  URL_to_Git_repo  optional_dir_rename

# update you submodules with origin/main or origin/your_branch
git submodule update --remote


# clone repo with submodules
git clone --recurse-submodules repo_url


# get submodules after clone
git submodule update --init
git submodule update --init --recursive  # if there are nested submodules

```


### Git config
```bash
# set default branch name
 git config --global init.defaultBranch main
```


### How to Push
When we clone a repo using HTTPS, how to push the repo using SSH. We have to set a new remote url. 
```bash
git remote set-url origin git@github.com:xxx/yyy.git
```

### References
[1] https://itsnotbugitsfeature.com/2019/10/22/splitting-a-big-pull-request-into-smaller-review-able-ones/