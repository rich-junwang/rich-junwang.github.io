---
title: "Basics of Web"
date: 2016-05-05T00:17:58+08:00
lastmod: 2016-05-05T00:17:58+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- Web
description: "Web tech"
weight:
slug: ""
draft: false # 是否为草稿
comments: true
reward: true # 打赏
mermaid: true #是否开启mermaid
showToc: true # 显示目录
TocOpen: true # 自动展开目录
hidemeta: false # 是否隐藏文章的元信息，如发布日期、作者等
disableShare: true # 底部不显示分享栏
showbreadcrumbs: true #顶部显示路径
cover:
    image: "images/web.jpg" #图片路径例如：posts/tech/123/123.png
    caption: "" #图片底部描述
    alt: ""
    relative: false
---


### Forward and Reverse Proxy Server
When clients (web browsers) make requests to sites and services on the Internet, the forward proxy server intercepts those requests and then communicates with servers on behalf of these clients like a middleman. 
![](images/forward_proxy.png)
Why we use forward proxy server?
- Circumvent restrictions. Sometimes restrictions through firewall are put on the access of the internet. A forward proxy can get around these restriction.
- Block contents. Proxies can be set up to block a certain type of user to access a specific type of online contents
- Protect online ID. 


A reverse proxy is an application that sits in front of back-end applications and forwards client (e.g. browser) requests to those applications. 
![](images/reverse_proxy.png)
Why we use reverse proxy server?
- Load balancing
- Protection from attacks
- Caching (such as CDN caching)
- SSL encryption

A simplified way to sum it up would be to say that a forward proxy sits in front of a client and ensures that no origin server ever communicates directly with that specific client. On the other hand, a reverse proxy sits in front of an origin server and ensures that no client ever communicates directly with that origin server.