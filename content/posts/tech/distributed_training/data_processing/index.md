---
title: "Data Processing in Distributed Training"
date: 2023-05-05T00:18:23+08:00
lastmod: 2023-05-05T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
- distributed training
description: "Distributed data processing"
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


A memory mapping is a region of the process’s virtual memory space that is mapped in a one-to-one correspondence with another entity. In this section, we will focus exclusively on memory-mapped files, where the memory of region corresponds to a traditional file on disk. For example, assume that the address 0xf77b5000 is mapped to the first byte of a file. Then 0xf77b5001 maps to the second byte, 0xf77b5002 to the third, and so on.

When we say that the file is mapped to a particular region in memory, we mean that the process sets up a pointer to the beginning of that region. The process can the dereference that pointer for direct access to the contents of the file. Specifically, there is no need to use standard file access functions, such as read(), write(), or fseek(). Rather, the file can be accessed as if it has already been read into memory as an array of bytes. Memory-mapped files have several uses and advantages over traditional file access functions:

- Memory-mapped files allow for multiple processes to share read-only access to a common file. As a straightforward example, the C standard library (glibc.so) is mapped into all processes running C programs. As such, only one copy of the file needs to be loaded into physical memory, even if there are thousands of programs running.
- In some cases, memory-mapped files simplify the logic of a program by using memory-mapped I/O. Rather than using fseek() multiple times to jump to random file locations, the data can be accessed directly by using an index into an array.
Memory-mapped files provide more efficient access for initial reads. When read() is used to access a file, the file contents are first copied from disk into the kernel’s buffer cache. Then, the data must be copied again into the process’s user-mode memory for access. Memory-mapped files bypass the buffer cache, and the data is copied directly into the user-mode portion of memory.
- If the region is set up to be writable, memory-mapped files provide extremely fast IPC data exchange. That is, when one process writes to the region, that data is immediately accessible by the other process without having to invoke a system call. Note that setting up the regions in both processes is an expensive operation in terms of execution time; however, once the region is set up, data is exchanged immediately.
- In contrast to message-passing forms of IPC (such as pipes), memory-mapped files create persistent IPC. Once the data is written to the shared region, it can be repeatedly accessed by other processes. Moreover, the data will eventually be written back to the file on disk for long-term storage.

<p align="center">
    <img alt="flat sharp minimum" src="images/mmap.png" width="60%" height=auto/> 
    <em>mmap file</em>
    <br>
</p>