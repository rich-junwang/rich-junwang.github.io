---
title: "SSH Connection"
date: 2022-01-11T00:18:23+08:00
lastmod: 2022-01-11T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- blog
description: "How to always keep ssh connection alive"
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

## Keep SSH Connected

There is always one issue that bothers me when using SSH to access server (e.g. EC2) which is that the ssh connection can disconnect very soon. I tried to make changes in the local ssh config: `~/.ssh/config`

```
Host remotehost
	HostName remotehost.com
	ServerAliveInterval 50
```

Then do a permission change
```
chmod 600 ~/.ssh/config
```

However, this doesn't work for me on Mac, and I don't know why. :( 

Then I tried to make changes on server side. 
In `/etc/ssh/sshd_config`, add or uncomment the following lines:
```
ClientAliveInterval 50
ClientAliveCountMax 10
```
Then restart or reload SSH server to help it recognize the configuration change
```
sudo service ssh restart  # for ubuntu linux
sudo service sshd restart  # for other linux dist
```

Finally, log out and try to login again
```
logout
```

This time it works! :)


## Adding SSH Public Key to Server
Adding ssh public key to server sometimes can make the connections eaiser. The command is simple:
```
cat ~/.ssh/id_ras.pub | ssh -i "my-keypair.pem"  ubuntu@myserver 'cat >> ~/.ssh/authorized_keys'
```