---
layout: post
title: Shell command for my reference
date: 2022-07-11
description: tricks and tips
tags: Tricks and Tips
categories: software
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