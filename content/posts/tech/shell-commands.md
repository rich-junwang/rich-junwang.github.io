---
title: "Shell Commands"
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

### Commonly Used Commands
```bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

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

```bash
$find . -type f -name "*.log" | xargs -n 1 echo rm
rm ./log/file5.log
rm ./log/file6.log
```
-n 1 argument, xargs turns each line into a command of its own.

-I option takes a string that gets replaced with the supplied input before the command executes. Commond choices are {} and %.
```bash
find ./log -type f -name "*.log" | xargs -I % mv % backup/
aws s3 ls --recursive s3://my-bucket/ | grep "my_test" | cut -d' ' -f4 | xargs -I{} aws s3 rm s3://my-bucket/{}
```

-P option specify the number of parallel processes used in executing the commands over the input argument list.

The command below parallelly encodes a series of wav files to mp3 format:
$find . -type f -name '*.wav' -print0 |xargs -0 -P 3 -n 1 mp3 -V8

When combining find with xargs, it's usually faster than using `exec` mentioned above.


#### Export and Xargs
To export each line as an environment variable we can use
```bash
export $(cat filename | xargs -L 1)
```


### rsync command
When use the following command, be careful about the relative path. In this command, we're using 4 processes. 
```bash
ls /my_model/checkpoints/source_dir | xargs -n1 -P4 -I% rsync -aP % target_dir

# the above command may suffer when there is only one folder inside the source dir that is too big. To solve this issue, use the following command. Note the -R here is relative path. 
cd src_dir && find . -type f -print0  | xargs -0 -P4 -I% rsync -avR % target_dir
```


### Tips and Tricks
1. Sometimes we need to copy multiple files from a directory. In order to copy multiple ones without explicitly listing all the absolute paths, we can use the following way. However, to use the autocomplete, we need to type left `{` first without the right one. 
```bash
cp /root/local/libs/{a.py, b.py} target_dir
```

### Sort and Cut
Sometimes we want to sort based on a specific field in string. We can use the following command
```
# cut 2nd field, from 2nd field split based on equal sign and sort based on the 3rd field
# cut field delimiter is -d, and sort field delimiter is -t
echo xxx | grep yyy | cut -d " " -f2 |  | sort -t = -k 3 -n | uniq | less
```

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



### Heredoc
In Bash and other shells like Zsh, a Here document (Heredoc) is a type of redirection that allows you to pass multiple lines of input to a command.

The syntax of writing HereDoc takes the following form:

```Bash
[COMMAND] <<[-] 'DELIMITER'
  HERE-DOCUMENT
DELIMITER
```

The first line starts with an optional command followed by the special redirection operator << and the delimiting identifier.
You can use any string as a delimiting identifier, the most commonly used are EOF or END.
If the delimiting identifier is unquoted, the shell will substitute all variables, commands and special characters before passing the here-document lines to the command.
Appending a minus sign to the redirection operator <<-, will cause all leading tab characters to be ignored. This allows you to use indentation when writing here-documents in shell scripts. Leading whitespace characters are not allowed, only tab.
The here-document block can contain strings, variables, commands and any other type of input.
The last line ends with the delimiting identifier. White space in front of the delimiter is not allowed.
Basic Heredoc Examples
In this section, we will look at some basic examples of how to use heredoc.

Heredoc is most often used in combination with the cat command .

In the following example, we are passing two lines of text containing an environment variable and a command to cat using a here document.

```
cat << EOF
The current working directory is: $PWD
You are logged in as: $(whoami)
EOF
```

As you can see from the output below, both the variable and the command output are substituted:

```
The current working directory is: /home/linuxize
You are logged in as: linuxize
```

Let’s see what will happen if we enclose the delimiter in single or double quotes.

```
cat <<- "EOF"
The current working directory is: $PWD
You are logged in as: $(whoami)
EOF
```

You can notice that when the delimiter is quoted no parameter expansion and command substitution is done by the shell.

```
The current working directory is: $PWD
You are logged in as: $(whoami)
```

If you are using a heredoc inside a statement or loop, use the <<- redirection operation that allows you to indent your code.

```
if true; then
    cat <<- EOF
    Line with a leading tab.
    EOF
fi
```

```
#output
Line with a leading tab.
```

Instead of displaying the output on the screen you can redirect it to a file using the >, >> operators.

```
cat << EOF > file.txt
The current working directory is: $PWD
You are logged in as: $(whoami)
EOF
```

If the file.txt doesn’t exist it will be created. When using > the file will be overwritten, while the >> will append the output to the file.

The heredoc input can also be piped. In the following example the sed command will replace all instances of the l character with e:

```
cat <<'EOF' |  sed 's/l/e/g'
Hello
World
EOF
```

```
#output
Heeeo
Wored
```

To write the piped data to a file:

```bash
cat <<'EOF' |  sed 's/l/e/g' > file.txt
Hello
World
EOF
```

Using Heredoc is one of the most convenient and easiest ways to execute multiple commands on a remote system over SSH .

When using unquoted delimiter make sure you escape all variables, commands and special characters otherwise they will be interpolated locally:

```bash
ssh -T user@host.com << EOF
echo "The current local working directory is: $PWD"
echo "The current remote working directory is: \$PWD"
EOF
```

```bash
#above command output
The current local working directory is: /home/linuxize
The current remote working directory is: /home/user
```


Another usage is to read strings into a variable. The following is an example where we can read a json format config to `my_config_json` variable. 
```bash
read -r -d '' my_config_json <<EOF
{
"working_dir": "${WORK_DIR}",
"env_vars": {
"PYTHONPATH": "${SRC_DIR}:$PYTHONPATH",
"LD_LIBRARY_PATH": "my_ld_path:$LD_LIBRARY_PATH"
}
}
EOF

# we can use the above config in Ray
ray job submit --address="http://127.0.0.1:8265" \
--runtime-env-json="${my_config_json}" \
-- python xx

```

Another example on heredoc:

```bash
# https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8b-104B/tr8b-104B-emb-norm-64n.slurm
config_json="./ds_config.$SLURM_JOBID.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "zero_allow_untested_optimizer": true,
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT
```

### Tmux
How to run a python script on each GPU
```bash
for rank in {0..7} ; do
CUDA_VISIBILE_DEVICES=$rank  tmux new-session -d -s my_session_${rank} python3 my_python_script my_arguments  2>&1 &
```

### References
[1] https://itsnotbugitsfeature.com/2019/10/22/splitting-a-big-pull-request-into-smaller-review-able-ones/