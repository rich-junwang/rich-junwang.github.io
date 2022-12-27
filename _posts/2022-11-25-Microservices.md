---
layout: post
title: Microservices
date: 2022-11-25
description: tricks and tips
tags: Tricks and Tips
categories: software
---

I'm learning setting up microservices on Udemy. In this blog, I'm documenting what I've learnt along the way. I'll keep updating this doc with progress of my study. 

## Data Management

The biggest challenge in microservices design is data management. To remove interdependency between services, in microservice system each service has its own database. This introduces the problem when we have a service C where its data is from service A database and service B database. Understanding of this problem will help us understand why we need message queue and redis in today's services design.







