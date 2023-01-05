---
layout: post
title: Shell command for my reference
date: 2022-07-11
description: tricks and tips
tags: Tricks and Tips
categories: software
---

### install on ec2
```
sudo yum install epel-release.noarch
sudo yum install xclip
```

### update python on ubuntu
When there are multiple version of python in the system, how to set the default python to use. Below we suppose to install newer version of python3.9
```
sudo apt install python3.9

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.[old-version] 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2

#after the following command, select the new version when being prompted and press enter
sudo update-alternatives --config python3

# There is a simpler way. From system admin perspective, this is not scalable.
sudo ln -sf /usr/bin/python3.9 /usr/bin/python3
```

### find a python package related files
```
pip show -f package-name
```

