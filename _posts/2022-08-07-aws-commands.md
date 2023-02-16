---
layout: post
title: AWS Commands
date: 2022-08-07
description: tricks and tips
tags: Tricks and Tips
categories: software
---
In this doc, I keep record of some commonly used aws related commands for my quick reference. I'll be very glad if this could be somewhat helpful to you.

### ECR 
ECR login

For aws-cli 2.7 or above version, use the command below:
```
# check all images
aws ecr describe-repositories

# login the docker
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<region>.amazonaws.com

# we can get the aws account id using the following command
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin "$(aws sts get-caller-identity --query Account --output text).dkr.ecr.<region>.amazonaws.com"


# pull the image
docker pull <image_name>

```



### GitLab
Gitlab  changes its authentication methods and the way it works is almost identical to Github. The easist way to use it is through personal token.


```
# For gitlab usage
# clone a repo using personal token
git clone https://oauth2:personal_token@gitlab.com/username/project.git

git remote set-url origin https://oauth2:personal_token@gitlab.com/username/project.git

git push https://personal_token@gitlab.com/username/project.git
```

```
## For github usage
git clone https://username:personal_token@github.com/username/project.git .

git remote set-url origin https://username:personal_token@github.com/username/project.git

git push https://personal_token@github.com/username/project.git
```


### Common AWS CLI
To get the current region,
```
aws configure get region
# if using the profile
aws configure get region --profile $PROFILE_NAME
```


### CloudWatch
To use cloudwatch insight, we can use the following query
```
fields @timestamp, @message, @logStream
| filter @logStream like /xxxxx/
| sort @timestamp desc
| limit 10000 
```