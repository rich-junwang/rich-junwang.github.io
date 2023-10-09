+++
title = "使用 Hugo 搭建个人网站（博客、个人主页）并发布到 Github 上"
date = 2020-10-04
lastmod = 2020-10-05T15:43:51+08:00
tags = ["Hugo", "Blog", "Website"]
categories = ["Hugo"]
draft = true
katex = true
+++

## 前言 {#前言}

相信很多人都有搭建个人网站的需求，可能是想写自己的博文，传达一些思想给社区。也有可能你是一名科研工作者，需要搭建个人学术主页展示科研成果。我也或多或少出于以上原因，选择搭建了个人网站，搭建过程中出现了许多问题，因此记录和分享，避免下次踩坑。Anyway，本文记录使用搭建个人博客的全过程，包括网站工具 [Hugo](https://gohugo.io/) 的介绍和使用设置，以及如何将个人网站发布在免费托管平台，也就是 GitHub Pages 上。

我选择 Hugo 的原因主要有三点：

-   简单易用；
-   [Hugo](https://gohugo.io/) 能够快速地构建个人网站；
-   拥有丰富的主题，可供挑选 [Hugo Themes](https://jamstackthemes.dev/ssg/hugo/)；

选择 GitHub Pages 的原因就更简单了，免费又好用。


## Hugo 的安装和使用 {#hugo-的安装和使用}

Hugo 宣传号称是世界上最快构建网站的框架，也是最流行的静态网站生成工具之一。


### 安装 Hugo {#安装-hugo}

由于我的操作系统是 MacOs 因此安装起来特别简单：

```shell
brew install hugo
```

其他平台可参考 [Hugo Install](https://gohugo.io/getting-started/installing)。


### 创建个人网站 {#创建个人网站}

```shell
hugo new site quickstart
```


### 使用 Hugo 主题 {#使用-hugo-主题}

我使用的是 [jane](https://github.com/xianmin/hugo-theme-jane), 将主题 `clone` 到 `theme` 目录下：

```shell
cd quickstart
git clone https://github.com/xianmin/hugo-theme-jane.git --depth=1 themes/jane
```

使用示例文本和默认的站点设置：

```shell
cp -r themes/jane/exampleSite/content ./
cp themes/jane/exampleSite/config.toml ./
```

启动 Hugo 服务器，在 [http://localhost:1313/](http://localhost:1313/) 将会看到示例 jane 主题的示例网站：

```shell
hugo server
```


### 个人配置和网站生成 {#个人配置和网站生成}

配置文件在网站根目录 `quickstart` 下 `config.toml` , 根据自身需求进行修改。在 jane 主题下的 `exampleSite` 文件夹中的文件可作为参考。默认的文章将存储在 `./content/post` 中，每当写完文章，运行 `hugo` 命令，Hugo 将自动生成静态网站到 `public` 文件夹中，我们只需要将该文件夹的内容发布在网络上即可。

更多关于主题的配置可以参考 [jane README.md](https://github.com/xianmin/hugo-theme-jane/blob/master/README-zh.md)。


## GitHub Pages {#github-pages}

我个人经常使用 GitHub，也见到很多大佬利用 GitHub pages 挂载自己的个人网站，发现配置起来也很简单，因此选择使用 GitHub pages 来进行配置，关于 GitHub pages 可以查看[官网](https://pages.github.com/)，主要包括四个步骤：

1.  创建一个与 `username` 同名的 **空** `username.github.io` 仓库，不包含任何内容，如 `readme.md=，比如我的用户名为 =kinredon`, 因此我创建了一个仓库，名为 `kinredon.github.io`;

2.  克隆仓库到本地

    ```shell
    git clone https://github.com/kinredon/kinredon.github.io
    ```

3.  添加个人网站内容到该仓库

    ```shell
    # copy 生成的网站内容到仓库文件夹下
    cp -rf quickstart/public/* kinredon.github.io/
    ```

4.  将文件内容同步更新到 GitHub 服务器上

    ```shell
    cd kinredon.github.io
    git add .
    git commit -m "init the website"
    git push
    ```

    此时，通过进入 [https://kinredon.github.io](https://kinredon.github.io) 即可访问自己的个人网站。

上面的步骤略显麻烦，每次需要将生成在 `public` 文件夹下的目录拷贝到 `kinredon.github.io` 目录下，然后发布到远程服务器。根据 [Host on GitHub](https://gohugo.io/hosting-and-deployment/hosting-on-github/)，发布到 GitHub Pages 有两种方式, 一种是直接使用仓库目录下的 `doc` 目录作为原本 `public` 目录，详情可以参考 [hugo 博客搭建](https://patrolli.github.io/xssq/post/hugo%5F%E5%8D%9A%E5%AE%A2%E6%90%AD%E5%BB%BA/)。我采用的方式是利用 GitHub Action 自动完成上述过程。


### 使用 GitHub Action 自动发布文章 {#使用-github-action-自动发布文章}

这里主要参考 [搭建个人blog](https://vinurs.me/posts/1a329bf3-fbb7-4006-9714-d3b072826376/), 使用 `master` 分支发布文章，使用一个新的 `source` 分支进行写作，写作完成后上传 `source` ，GitHub Action 自动将 `source` 分支的 `publish` 文件夹拷贝到 `master` 分支，从而完成文章的发布。

主要步骤如下：

1.  在 GitHub 上的个人网站仓库 `kinredon.github.io` 新建 `source` 分支

    {{< figure src="/ox-hugo/pngpaste_clipboard_file_20211004154805.png" caption="Figure 1: 创建 source 分支，由于我已经创建过，所以这里以 source-1 为例" >}}

2.  清除文件夹 `kinredon.github.io` 中的内容，并将个人网站 `quickstart` 中的所有内容 copy 到 `kinredon.github.io` ：

    ```shell
    git clone --branch=source https://github.com/kinredon/kinredon.github.io.git
    rm -rf kinredon.github.io/*
    cp -rf quickstart/* kinredon.github.io
    ```

3.  生成 `ACTIONS_DEPLOY_KEY`

    ```shell
    ssh-keygen -t rsa -b 4096 -C "$(git config user.email)" -f gh-pages -N ""
    ```

    将生成的私钥文件 `gh-pages` (注意不是公钥 `gh-pages.pub`) 中的内容复制填写到 GitHub 仓库设置中，即在 `kinredon.github.io` 项目主页中，找到 Repository Settings -> Secrets -> 添加这个私钥的内容并命名为 `ACTIONS_DEPLOY_KEY` 。
    然后在 `kinredon.github.io` 项目主页中，找到 Repository Settings -> Deploy Keys -> 添加这个公钥的内容，命名为 `ACTIONS_DEPLOY_KEY` ，并勾选 Allow write access。

4.  配置 workflow

    创建 workflow 文件

    ```shell
    mkdir -p .github/workflows/
    touch .github/workflows/main.yml
    ```

    在 `main.yaml` 中撰写 workflow，内容如下：

    ```shell
    name: github pages

    on:
      push:
        branches:
    ​    - source

    jobs:
      build-deploy:
        runs-on: ubuntu-18.04

        steps:
    ​    - uses: actions/checkout@master
    ​    - name: Checkout submodules
          shell: bash
          run: |
            auth_header="$(git config --local --get http.https://github.com/.extraheader)"
            git submodule sync --recursive
            git -c "http.extraheader=$auth_header" -c protocol.version=2 submodule update --init --force --recursive --depth=1

        - name: Setup Hugo
          uses: peaceiris/actions-hugo@v2
          with:
            hugo-version: 'latest'
            extended: true

        - name: Build
          run: hugo --gc --minify --cleanDestinationDir

        - name: Deploy
          uses: peaceiris/actions-gh-pages@v3
          with:
            deploy_key: ${{ secrets.ACTIONS_DEPLOY_KEY }}
            publish_dir: ./public
            publish_branch: master

    ```

    注意，如果你的仓库是 `master` 分支作为主分支，将 `publish_branch` 后面的 `main` 修改为 `master` ;

5.  将 source 分支发送到远程

    发送脚本 `deploy.sh` :

    ```shell
    #!/bin/bash
    git add .
    git commit -m "update article"
    git push
    ```

    推送到远程分支：

    ```shell
    sh deploy.sh
    ```

    推送成功后，可以在项目主页中的 action 里面查看自动部署是否成功，即 <https://github.com/kinredon/kinredon.github.io/actions>；