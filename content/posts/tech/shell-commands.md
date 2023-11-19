---
title: "Shell Commands"
date: 2022-02-11T00:18:23+08:00
lastmod: 2022-02-11T00:18:23+08:00
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
reward: true # 打赏
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

### Find Command
The original post is here: https://www.baeldung.com/linux/find-exec-command

#### 1. Basics
The find command is comprised of two main parts, the expression and the action.
When we initially use find, we usually start with the expression part. This is the part that allows us to specify a filter that defines which files to select.

A classic example would be:

```
$ find Music/ -name *.mp3 -type f
Music/Gustav Mahler/01 - Das Trinklied vom Jammer der Erde.mp3
Music/Gustav Mahler/02 - Der Einsame im Herbst.mp3
```

The action part in this example is the default action, -print. This action prints the resulting paths with newline characters in between. It’ll run if no other action is specified.

In contrast, the -exec action allows us to execute commands on the resulting paths.
Let’s say we want to run the file command on the list of mp3 files we just found to determine their filetype. We can achieve this by running the following command:

```
$ find Music/ -name *.mp3 -exec file {} \;
Music/Gustav Mahler/01 - Das Trinklied vom Jammer der Erde.mp3:
  Audio file with ID3 version 2.4.0, contains:MPEG ADTS, layer III, v1, 128 kbps, 44.1 kHz, Stereo
```

Let’s dissect the arguments passed to the -exec flag, which include:
A command: file \
A placeholder: {} \
A command delimiter: \; \
Now we’ll walk through each of these three parts in-depth.

#### 2. The Command
Any command that can be executed by our shell is acceptable here. We should note that this isn’t our shell executing the command, rather we’re using Linux’s exec directly to execute the command. This means that any shell expansion won’t work here, as we don’t have a shell. Another effect is the unavailability of shell functions or aliases.

As a workaround for our missing shell functions, we can export them and call bash -c with our requested function on our file. To see this in action, we’ll continue with our directory of Mahler’s mp3 files. Let’s create a shell function that shows the track name and some details about the quality:
```
function mp3info() {
    TRACK_NAME=$(basename "$1")
    FILE_DATA=$(file "$1" | awk -F, '{$1=$2=$3=$4=""; print $0 }')
    echo "${TRACK_NAME%.mp3} : $FILE_DATA"
}
```

If we try to run the mp3info command on all of our files, -exec will complain that it doesn’t know about mp3info:
```
find . -name "*.mp3" -exec mp3info {} \;
find: ‘mp3info’: No such file or directory
```
As mentioned earlier, to fix this, we’ll need to export our shell function and run it as part of a spawned shell:
```
$ export -f mp3info
$ find . -name "*.mp3" -exec bash -c "mp3info \"{}\"" \;
01 - Das Trinklied vom Jammer der Erde :      128 kbps  44.1 kHz  Stereo
02 - Der Einsame im Herbst :      128 kbps  44.1 kHz  Stereo
03 - Von der Jugend :      128 kbps  44.1 kHz  Stereo
```
Note that because some of our file names hold spaces, we need to quote the results placeholder.

#### 3. The Results Placeholder
The results placeholder is denoted by two curly braces {}.

We can use the placeholder multiple times if necessary:

```
find . -name "*.mp3" -exec bash -c "basename \"{}\" && file \"{}\" | awk -F: '{\$1=\"\"; print \$0 }'" \;
01 - Das Trinklied vom Jammer der Erde.mp3
  Audio file with ID3 version 2.4.0, contains MPEG ADTS, layer III, v1, 128 kbps, 44.1 kHz, Stereo
02 - Der Einsame im Herbst.mp3
  Audio file with ID3 version 2.4.0, contains MPEG ADTS, layer III, v1, 128 kbps, 44.1 kHz, Stereo
03 - Von der Jugend.mp3
  Audio file with ID3 version 2.4.0, contains MPEG ADTS, layer III, v1, 128 kbps, 44.1 kHz, Stereo
```

In the above example, we ran both the basename, as well as the file commands. To allow us to concatenate the commands, we spawned a separate shell, as explained above.

#### 4. The Delimiter
We need to provide the find command with a delimiter so it’ll know where our -exec arguments stop. Two types of delimiters can be provided to the -exec argument: the semi-colon(;) or the plus sign (+). As we don’t want our shell to interpret the semi-colon, we need to escape it (\;).

The delimiter determines the way find handles the expression results. If we use the semi-colon (;), the -exec command will be repeated for each result separately. On the other hand, if we use the plus sign (+), all of the expressions’ results will be concatenated and passed as a whole to the -exec command, which will run only once.

Let’s see the use of the plus sign with another example:

```
$ find . -name "*.mp3" -exec echo {} +
./Gustav Mahler/01 - Das Trinklied vom Jammer der Erde.mp3 ./Gustav Mahler/02 -
  Der Einsame im Herbst.mp3 ./Gustav Mahler/03 - Von der Jugend.mp3 ./Gustav Mahler/04 -
  Von der Schönheit.mp3 ./Gustav Mahler/05 - Der Trunkene im Frühling.mp3
  ./Gustav Mahler/06 - Der Abschied.mp3
```

When running echo, a newline is generated for every echo call, but since we used the plus-delimiter, only a single echo call was made. Let’s compare this result to the semi-colon version:
```
$ find . -name "*.mp3" -exec echo {} \;
./Gustav Mahler/01 - Das Trinklied vom Jammer der Erde.mp3
./Gustav Mahler/02 - Der Einsame im Herbst.mp3
```
From a performance point of view, we usually prefer to use the plus-sign delimiter, as running separate processes for each file can incur a serious penalty in both RAM and processing time.

However, we may prefer using the semi-colon delimiter in one of the following cases:

- The tool run by -exec doesn’t accept multiple files as an argument.
- Running the tool on so many files at once might use up too much memory.
- We want to start getting some results as soon as possible, even though it’ll take more time to get all the results.

One of the commands I use often with is 
```
find my_directory/  -type f -exec lfs hsm_restore {} \; 
```


### Xargs Command
There are commands that only take input as arguments like `cp`, `rm`, `echo` etc. We can use xargs to convert input coming from standard input to arguements.

```
$find . -type f -name "*.log" | xargs -n 1 echo rm
rm ./log/file5.log
rm ./log/file6.log
```
-n 1 argument, xargs turns each line into a command of its own.

-I option takes a string that gets replaced with the supplied input before the command executes. Commond choices are {} and %.
```
find ./log -type f -name "*.log" | xargs -I % mv % backup/
aws s3 ls --recursive s3://my-bucket/ | grep "my_test" | cut -d' ' -f4 | xargs -I{} aws s3 rm s3://my-bucket/{}
```

-P option specify the number of parallel processes used in executing the commands over the input argument list.

The command below parallelly encodes a series of wav files to mp3 format:
$find . -type f -name '*.wav' -print0 |xargs -0 -P 3 -n 1 mp3 -V8

When combining find with xargs, it's usually faster than using `exec` mentioned above.


### rsync command
When use the following command, be careful about the relative path. In this command, we're using 16 processes. 
```bash
ls /my_model/checkpoints/source_dir | xargs -n16 -P -I% rsync -aP % target_dir
```


### Here Document/Text



### Tips and Tricks
1. Sometimes we need to copy multiple files from a directory. In order to copy multiple ones without explicitly listing all the absolute paths, we can use the following way. However, to use the autocomplete, we need to type left `{` first without the right one. 
```bash
cp /root/local/libs/{a.py, b.py} target_dir
```

