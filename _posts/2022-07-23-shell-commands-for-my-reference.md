---
layout: post
title: Shell command for my reference
date: 2022-07-11
description: tricks and tips
tags: Tricks and Tips
categories: software
---

### install on ec2
```
sudo yum install epel-release.noarch
sudo yum install xclip
```

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
git rebase master -X theirs
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