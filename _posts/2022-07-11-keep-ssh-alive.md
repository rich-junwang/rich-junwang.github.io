There is always one issue that bothers me when ssing SSH to access server (e.g. EC2) which is that the ssh connection can disconnect very soon. I tried to make changes in the local ssh config: ~/.ssh/config

```
Host remotehost
	HostName remotehost.com
	ServerAliveInterval 50
```

Then do a permission change
```
chmod 600 ~/.ssh/config
```

However, this doesn't work for me on Mac, and I don't know why. Then I tried to make changes on server side. 
In `/etc/ssh/sshd_config
```s time it works! :)
