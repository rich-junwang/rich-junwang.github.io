---
layout: post
title: Keep SSH Connected
date: 2022-07-11
description: tricks and tips
tags: Tricks and Tips
categories: software
---

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