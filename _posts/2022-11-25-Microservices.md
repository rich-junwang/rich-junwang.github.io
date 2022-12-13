---
layout: post
title: Microservices
date: 2022-11-25
description: tricks and tips
tags: Tricks and Tips
categories: software
---

## Data Management

The biggest challenge in microservices design is data management. To remove interdependency between services, in microservice system each service has its own database. This introduces the problem when we have a service C where its data is from service A database and service B database. Understanding of this problem will help us understand why we need message queue and redis in today's services design.







