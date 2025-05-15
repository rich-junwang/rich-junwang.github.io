---
title: "Remote Development with VSCode"
date: 2020-02-08T12:01:14-07:00
lastmod: 2020-02-08T12:01:14-07:00
author: ["Jun"]
keywords: 
- 
categories: # 没有分类界面可以不填写
- 
tags: # 标签
- Tool
description: "How to use vscode for remote development"
weight:
slug: ""
draft: false # 是否为草稿
comments: true # 本页面是否显示评论
reward: false # 打赏
mermaid: true #是否开启mermaid
showToc: true # 显示目录
TocOpen: true # 自动展开目录
hidemeta: false # 是否隐藏文章的元信息，如发布日期、作者等
disableShare: true # 底部不显示分享栏
showbreadcrumbs: true #顶部显示路径
cover:
    image: "" #图片路径例如：posts/tech/123/123.png
    zoom: # 图片大小，例如填写 50% 表示原图像的一半大小
    caption: "" #图片底部描述
    alt: ""
    relative: false
---

### Remote Development

I used to do development on my local machine, and use **fswatch** and **rsync** to sync changes to server in real time. It works perfectly when development dependencies are simple and easy to set up. Generally I refer this development mode as local development. However, as more and more development environments are containerized, it becomes non-trivial to set up environment everytime. Recently, I started using VSCode as it has better support to leverage remote server development environment. 


One great feature of VScode is that it works well with docker and kubernetes, i.e. we can attach VSCode to docker or kubernetes pods easily. In vscode terminal, we can execute commands like we're doing on the server. I call this kind of development as remote development. 


One problem with remote development is that we can't save our changes locally. Once server dies, all our changes are gone. The solution is to use git. Since docker doesn't come with an editor, when we use git, we have to set vscode as the editor:
```
git config --global core.editor "code --wait"
```

Another issue is we have to install extensions on remote server. For instance, we have to install python extension in order to use python interpreter in remote mode. 

VScode also has a [nice extension tool](https://marketplace.visualstudio.com/items?itemName=Natizyskunk.sftp) to sync code to remote server.

### VScode Shortcuts
If it's on Mac, replace CTRL key with CMD key

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


(11) open a new window for a new project
CTRL + SHIFT + N


(12) Open terminal
CTRL + ` to toggle terminal panel. Note one mac here it's CTRL as well.

(13) Close search window after global search
In `keybindings.json` add the following lines
```
{
  "key": "Escape",
  "command": "workbench.view.explorer",
  "when": "searchViewletVisible"
}
```

(14) How to open unlimited number of tabs
In `settings.json` add the following key-value pair:
```
"workbench.editor.enablePreview": false
```

### Config
Here is the config settings I used.
```json
{
    "editor.fontSize": 16,
    "editor.fontFamily": "Monaco, 'Courier New', monospace",
    "editor.wordWrap": "on",
    "editor.tabCompletion": "on",
    "editor.tabSize": 4,
    "editor.suggest.snippetsPreventQuickSuggestions": false,
    "gitlens.codeLens.authors.enabled": false,
    "git.timeline.showAuthor": false,
    "gitlens.codeLens.recentChange.enabled": false,
    "gitlens.codeLens.enabled": false,
    "gitlens.currentLine.enabled": false,
    "gitlens.currentLine.pullRequests.enabled": false,
    "redhat.telemetry.enabled": false,
    "terminal.integrated.fontSize": 16,
    "editor.minimap.enabled": false,
    "python.terminal.activateEnvironment": false,
    "workbench.editor.enablePreview": false,
    "[python]": {
        "editor.formatOnType": true
    },
    "settingsSync.ignoredSettings": [],
    "settingsSync.ignoredExtensions": [],
      //失去焦点后自动保存
    "files.autoSave": "onFocusChange",
    "terminal.integrated.inheritEnv": false,
    "gitlens.codeLens.recentChange.enabled": false,
    "gitlens.blame.avatars": false,
    "gitlens.blame.format": "${message|20?} ${agoOrDate|14-}",
    "gitlens.hovers.avatars": false,
    "json.format.enable": true,
    "files.associations": {
        "*.jsonl": "json"
    },
    "update.showReleaseNotes": false,
    "workbench.editor.revealIfOpen": true
}
```



### Remote SSH config
Install the plugin from [here](https://github.com/Natizyskunk/vscode-sftp?tab=readme-ov-file)

Connecting to a single serve the `sftp.json` is like this:
```json
{
    "name": "Profile1",
    "host": "42-32-123-45.mycompute.com",
    "protocol": "sftp",
    "port": 22,
    "secure": true,
    "username": "ubuntu",
    "remotePath": "/home/ubuntu/myproject", // <--- This is the path which will be downloaded if you "Download Project"
    "password": "password",
    "openSsh": false,
    "context": "my_local_directory_on_mac", // source dir
    "privateKeyPath": "/Users/xxx/.ssh/id_rsa",  // needed when use passwordless ssh
    "uploadOnSave": true,
    "useTempFile": false,
    "ignore": [
      "**/.vscode/**",
      "**/.git/**",
      "**/.DS_Store"
    ]
}
```

Connecting to multiple servers, the "sftp.json" is like this:
```json
{
    "protocol": "sftp",
    "port": 22,
    "secure": true,
    "remotePath": "/home/ubuntu/myproject", // <--- This is the path which will be downloaded if you "Download Project"
    "openSsh": false,
    "context": "my_local_directory_on_mac", // source dir
    "privateKeyPath": "/Users/xxx/.ssh/id_rsa",  // needed when use passwordless ssh
    "uploadOnSave": false,
    "useTempFile": false,
    "ignore": [
      "**/.vscode/**",
      "**/.git/**",
      "**/.DS_Store"
    ],
    "profiles": {
      "my_server1":{
        "name": "Profile1",
        "host": "42-32-123-45.mycompute.com",
        "username": "ubuntu",
        "password": "password"
      },
      "my_server2":{
        "name": "Profile2",
        "host": "42-32-123-46.mycompute.com",
        "username": "ubuntu",
        "password": "password"
      },
      "my_server3":{
        "name": "Profile3",
        "host": "42-32-123-47.mycompute.com",
        "username": "ubuntu",
        "password": "password"
      }
    }
}
```

After saving the `sftp.json` in the `.vscode` folder, we can trigger a start sync. 
- CTRL + P to start command palette
- using `>` to start a command, typing sftp. Select for example sync from local to remote.


### Paste and Indentation
Install `Paste and Indent` from g3rry
Adding the following to keybindings json
```
{
    "key": "ctrl+v",
    "command": "pasteAndIndent.action",
    "when": "editorTextFocus && !editorReadonly"
},
{
    "key": "ctrl+v",
    "command": "editor.action.clipboardPasteAction",
    "when": "!editorTextFocus"
},
{
    "key": "ctrl+shift+v",
    "command": "editor.action.clipboardPasteAction",
    "when": "editorTextFocus && !editorReadonly"
}
```
## References
[1] https://code.visualstudio.com/shortcuts/keyboard-shortcuts-linux.pdf