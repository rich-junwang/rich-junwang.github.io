---
title: "A Walk in the Cloud"
date: 2022-05-05T00:17:58+08:00
lastmod: 2022-05-05T00:17:58+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- tech
description: "AWS"
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
In this doc, I keep record of some commonly used aws related commands for my quick reference. I'll be very glad if this could be somewhat helpful to you.

### ECR 
ECR login

For aws-cli 2.7 or above version, use the command below:
```bash
# check all images
aws ecr describe-repositories

# login the docker
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<region>.amazonaws.com

# we can get the aws account id using the following command
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin "$(aws sts get-caller-identity --query Account --output text).dkr.ecr.<region>.amazonaws.com"


# pull the image
docker pull <image_name>

# If we pushed the image using sudo, then pull also add sudo
sudo docker pull <image_name>
```

Sometimes we need one image in one region, but it's pushed to another region. We can do the dollowing steps to push the image to target region.
```bash
# login to the region where the image current is. Here assume it's in us-east-1
REGION=us-east-1 ; aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.${REGION}.amazonaws.com

# Then pull the image from ECR
docker pull image_name

# Find out the image id
docker image ls | grep image_name | cut -f3 

# tag
docker tag  image_id new_image_tag_with_new_region

# login to the new region
REGION=us-west-2 ; aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.${REGION}.amazonaws.com

# push
docker push new_image_tag_with_new_region
```

### GitLab and Github
Gitlab  changes its authentication methods and the way it works is almost identical to Github. The easiest way to use it is through personal token.


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

Another simpler way to clone personal repo on a new machine
(1) Add ssh public key into github/gitlab webpage. settings -> ssh and GPG keys -> add ssh key.
(2) To clone a repo, use ssh link. git@github.com:xxx 

We can also use https to clone a repo, but need to add personal access token from github.


### Common AWS CLI
To get the current region,
```bash
aws configure get region
# if using the profile
aws configure get region --profile $PROFILE_NAME

# aws sync with exclude
aws s3 sync s3://my-first-bucket s3://my-second-bucket --exclude 'datasets/*'


# get the identity
aws sts get-caller-identity


# export credentials
export $(ada credentials print --account xxx --role myrole --provider=myprovider --profile my_profile --format env | xargs -L 1)
```


### CloudWatch
To use cloudwatch insight, we can use the following query
```bash
fields @timestamp, @message, @logStream
| filter @logStream like /xxxxx/
| sort @timestamp desc
| limit 10000 
```


### VPC and Security Group
Security group controls how we login the instance (like through ssh etc)
VPC determines what kind of resource we can visit from the instance. For instance if we are able to access specific EFS and FSx.
Private VPC subnet will require a bastion to connect to instance.

<p align="center">
    <img alt="VPC" src="images/vpc.png" width="80%"/>
    kubernetes architecture, image from [1]
</p>

An Internet Gateway is a logical connection between a VPC and the Internet. If there is no Internet Gateway, then the VPC has no direct access to the Internet. (However, Internet access might be provided via a Transit Gateway, which itself would need an Internet Gateway.)

Think of the Internet Gateway as the wire that you use to connect your home router to the Internet. Pull out that wire and your home network won't be connected to the Internet.

A subnet is a 'public subnet' if it has a Route Table that references an Internet Gateway.

A NAT Gateway receives traffic from a VPC, forwards it to the Internet and then returns the response that was received. It must live in a public subnet because it needs to communicate with the Internet (and therefore needs a route to the Internet Gateway).

Resources in a private subnet (which, by definition, cannot route to the Internet Gateway) will have their Internet-bound requests sent to the NAT Gateway (due to a Route Table configuration). The NAT Gateway will then forward that request to the Internet and return the response that was received from the Internet.

NAT Gateways exist because organizations want the additional security offered by private subnets, which guarantee that there is no inbound access from the Internet. Similar security can be provided with a Security Group, so private subnets aren't actually required. However, people who are familiar with traditional (non-cloud) networking are familiar with the concept of public and private subnets, so they want to replicate that architecture in the cloud. Physical network routers only apply rules at the boundary of subnets, whereas Security Groups can be applied individually to each Resource. It's a bit like giving each resource its own router.

You are right that all of the above is implemented as a virtual network. There is no physical device called an Internet Gateway or a NAT Gateway. Much of it is logical routing, although the NAT Gateway does involve launching infrastructure behind-the-scenes (probably on the same infrastructure that runs EC2 instances). The NAT Gateway only connects to one VPC -- it is not a 'shared service' like Amazon S3, which is available to many AWS users simultaneously.

You also mention performing work 'manually'. An entire VPC (including subnets, route tables, Internet Gateway, NAT Gateway, Security Groups) can be deployed automatically using an AWS CloudFormation template, or via the VPC Wizard in the VPC management console.


### EFS

```bash
# install efs-utils, depends on platform, it could be different.
# Then
mkdir $HOME/efs
sudo mount -t efs efs_id $HOME/efs

```


### EKS
```bash
# create an eks without node group
eksctl create cluster --region us-east-1 --without-nodegroup --vpc-public-subnets ${subnets} 


# dry run
eksctl create cluster -f ./eks.yaml --dry-run
```


When use cloudformation to create eks node group, the hardware could fail. We can use the following command to check the failure. The first failure in the stack (the resource provision is in a certain order) is usually what we're looking for. 
```bash
aws cloudformation describe-stack-events --stack-name my-cluster-stack-name
```

To delete a failed stack
```bash
aws cloudformation delete-stack --stack-name my-cluster-stack-name
```



### FSx for Lustre
Lustre (linux + cluster) is a high performance file system. FSx for lustre is an aws managed service to launch and run for lustre file system.


#### FSx and S3
s3 is long-term durable storage. FSx for lustre is used when processing data. Usually FSx is used as a high performance file system linked into s3 bucket. 
 
1. Store your data on s3
2. Create an FSx file system and link it to your s3 bucket
3. At any time, use a lustre command to write changes back to s3
4. Delete FSx file system when you're done processing
