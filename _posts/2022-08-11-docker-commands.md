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
nvidia-docker run --entrypoint /bin/bash --rm -it --name my_container_name  image_name

# mount a volume to docker
nvidia-docker run --entrypoint /bin/bash -v $PWD/transforms_cache:/transforms_cache --rm -it image_name

# add env to docker system
nvidia-docker run --entrypoint /bin/bash -v $PWD/transforms_cache:/transforms_cache --rm --env SM_CHANNEL_TRAIN=/opt/ml/input/data/train -it image_name

```


### Check all containers
```
docker ps -a
```


### Clean space
```
docker rmi -f $(docker images -a -q)
sudo docker system prune
```


### Install package
Install packages inside a running docker. Usually we're able to install package based on distributeion of linux system running in the docker. For example, if it's ubuntu, then the command is 
```
apt-get -y update
apt-get -y install tmux  # package name
```




