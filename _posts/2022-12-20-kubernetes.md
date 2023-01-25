---
layout: post
title: Kubernetes
date: 2022-12-20
description: tricks and tips
tags: Software
categories: Software
---



### Basics

Kubernetes, also known as “k8s”, is an open source platform solution provided by Google for scheduling and automating the deployment, management, and scaling of containerized applications. Kubernetes has the ability of scheduling and running application containers on a cluster of physical or virtual machines. In managing the applications, the concepts of 'labels' and 'pods' are used to group the containers which make up an application. Currently, it supports Docker for containers.

<p align="center">
    <img alt="gopher dataset" src="/assets/img/kubernetes.png" width="80%"/>
    <br>
    <em>kubernetes architecture, image from [1]</em>
    <br>
</p>


### Basic Operations
To find out all the pods, using the following command
```
kubectl get pods 
kubectl get pods | grep username 
```

We can use `kubectl` to copy files to/from the pod. Be careful that your container may not support `~` this kind of path expansion.
```
kubectl cp src_file_path pod:dest_file_path
```

To use rsync is not that straightforward, I'm using the tool from [here](https://serverfault.com/questions/741670/rsync-files-to-a-kubernetes-pod).
```
# save the file as krsync

#!/bin/bash

if [ -z "$KRSYNC_STARTED" ]; then
    export KRSYNC_STARTED=true
    exec rsync --blocking-io --rsh "$0" $@
fi

# Running as --rsh
namespace=''
pod=$1
shift

# If use uses pod@namespace rsync passes as: {us} -l pod namespace ...
if [ "X$pod" = "X-l" ]; then
    pod=$1
    shift
    namespace="-n $1"
    shift
fi

exec kubectl $namespace exec -i $pod -- "$@"
```

Then use the following command to sync files. Note that you have to install `rsync` on the pod. 
```
krsync -av --progress --stats src-dir/ pod:/dest-dir

# with namespace
krsync -av --progress --stats src-dir/ pod@namespace:/dest-dir

```
Sometimes we have to change file ownership. Check out more [here](https://vhs.codeberg.page/post/recover-files-kubernetes-persistent-volume/)

```
chown -R 33:33 /data/uploads
```




## References
[1] [Setting up a Kubernetes cluster using Docker in Docker](https://callistaenterprise.se/blogg/teknik/2017/12/20/kubernetes-on-docker-in-docker/) <br>

