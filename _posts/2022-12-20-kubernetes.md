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
kubectl get pods -n my_namespace_name  # get pod understand a namespace
```

To get all the containers running the pod, using the following command
```
kubectl get pods my_pod_name -o custom-columns='NAME:.metadata.name,CONTAINERS:.spec.containers[*].name'
kubectl describe pod my_pod_name  -n my_namespace_name
```

View logs of job running in the pod
```
kubectl logs my_pod_name
kubectl attach my_pod_name  # works with tqdm 
```

Log into the pod
```
kubectl exec -it my_pod_name -- /bin/bash
```


We can use `kubectl` to copy files to/from the pod. Be careful that your container may not support `~` this kind of path expansion.
```
kubectl cp src_file_path pod:dest_file_path
```

To use rsync is not that straightforward, I'm using the tool from [here](https://serverfault.com/questions/741670/rsync-files-to-a-kubernetes-pod).
```
# save the file as krsync, and put it to /usr/bin, and chmod +x to the file

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
To make it easier to use, we can add the following to the .zshrc file
```
function krsync_watch_and_sync_to {
        fswatch -o . | xargs -n1 -I{} krsync -av --progress --stats *(D)  $1
}
```

Sometimes we have to change file ownership. Check out more [here](https://vhs.codeberg.page/post/recover-files-kubernetes-persistent-volume/)

```
chown -R 33:33 /data/uploads
```




## References
[1] [Setting up a Kubernetes cluster using Docker in Docker](https://callistaenterprise.se/blogg/teknik/2017/12/20/kubernetes-on-docker-in-docker/) <br>
[2] https://kubernetes.io/docs/reference/kubectl/cheatsheet/ <br>

