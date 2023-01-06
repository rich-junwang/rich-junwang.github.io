---
layout: post
title: Remote development with VSCode
date: 2022-07-04
description: make vscode faster
tags: Tricks and Tips
categories: software
---

### Remote Development

I used to do development on my local machine, and use **fswatch** and **rsync** to sync changes to server in real time. It works perfectly when development dependencies are simple and easy to set up. Generally I refer this development mode as local development. However, as more and more development environments are containerized, it becomes non-trivial to set up environment everytime. Recently, I started using VSCode as it has better support to leverage remote server development environment. 


One great feature of VScode is that it works well with docker and kubernetes, i.e. we can attach VSCode to docker or kubernetes pods easily. In vscode terminal, we can execute commands like we're doing on the server. I call this kind of development as remote development. 

One problem with remote development is that we can't save our changes locally. Once server dies, all our changes are gone. The solution is to use git. Since docker doesn't come with an editor, when we use git, we have to set vscode as the editor:
```
git config --global core.editor "code --wait"
```


### VScode Shortcuts

(1) CTRL + X : cut a line \
(2) duplicate a line: duplicate can be achieved by CTRL+C and CTRL+V with cursor in the line (without selection)

(3) edit multiple line

* edit multiple line simultaneously: CTRL + SHIFT + up/down arrow. (This will be continuous)
* ALT + CLICK: can select multiple place and edit
* CTRL + SHIFT + L:  edit all variable in the file

(4) block comment: select the block, then CTRL + SHIFT + A

(5) line comment: CTRL + /

(6) search & replace

* single file search: CTRL + F
* single file replace: CTRL + H
* global search: CTRL + SHIFT + F
* global replace: CTRL + SHIFT + H

(7) move a line upward or downward: ALT + up/down arrow

(8) select a line: CTRL + L

(9) palette

* open palette: CTRL + P
* \>: type command
* @: find symbol
* #: find all relevant ones
* : go to line


(10) split screen
vertical split: CTRL + \




## References
[1] https://code.visualstudio.com/shortcuts/keyboard-shortcuts-linux.pdf