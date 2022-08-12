---
layout: post
title: Docker Commands
date: 2022-08-11
description: tricks and tips
tags: Tricks and Tips
categories: software
---
In the last blog, we talked about commonly used AWS commands. In this blog, I'll document some commonly used docker commands to save some time when I need them.

### Image
```
docker image ls
```

### Run docker container
```
# rm is to clean constainer after exit
# it is interactive tty
nvidia-docker run --entrypoint /bin/bash --rm -it image_name


nvidia-docker run --entrypoint /bin/bash -v $PWD/transforms_cache:/transforms_cache --rm -it image_name
```


### Check all containers
```
docker ps -a
```


### Clean space
```
docker rmi -f $(docker image -a -q)
sudo docker system prune
```





