---
title: "Docker Commands"
date: 2021-06-18T00:18:23+08:00
lastmod: 2021-07-18T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
    - docker
description: "Docker"
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
In the last blog, we talked about commonly used AWS commands. In this blog, I'll document some commonly used docker commands to save some time when I need them. Images defines what the container is. Container is the actually running virtual machine.


### Docker setup
```bash
# check docker status
systemctl show --property ActiveState docker

# if it's inactive, then start the docker daemon
sudo systemctl start docker
```


### Image
```bash
# list all images
docker image ls
# list all containers
docker container ls
# stop/remove a container
docker container stop container_id
docker container rm container_id

# pull an image
docker pull iamge_name
```

### Run docker container
```
# rm is to clean constainer after exit
# it is interactive tty
# for normal docker image
docker run --entrypoint /bin/bash -it <image_name> 
# for nvidia docker image
nvidia-docker run --entrypoint /bin/bash --rm -it --name my_container_name  image_name

# mount a volume to docker
# --rm delete docker on exit
nvidia-docker run --entrypoint /bin/bash -v $PWD/transforms_cache:/transforms_cache --rm -it image_name

# add env to docker system
nvidia-docker run --entrypoint /bin/bash -v $PWD/transforms_cache:/transforms_cache --rm --env SM_CHANNEL_TRAIN=/opt/ml/input/data/train -it image_name

# docker run to use GPU, we can use another command
docker run --entrypoint /bin/bash --gpus all -it xxxx_image_name
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


### Docker build
We can use the following command to build docker image. Notice that the path is . (current directory). The path (a set of files) is called context and files inside can be used in `COPY` command in dockerfile. In building process, context will be packed into a tar file. So it's good to put unnecessary files into `.dockerignore` file and select a reasonable path as context.
```
docker build -f Dockerfile_my_docker -t ${TAG} . --build-arg REGION=${region}
```


### Login AWS ECR
```
REGION=us-east-1 ; aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/ 

# sometimes we can get errors like `no basic auth credentials`, simple method is to remove docker config file
rm .docker/config.json

# Due to some reason, the docker is not able to update the config file.
```
