---
title: "Add Math Equation in Blog"
date: 2021-06-18T00:18:23+08:00
lastmod: 2021-07-18T00:18:23+08:00
author: ["Jun"]
keywords: 
- 
categories: 
- 
tags: 
    - Web
description: "Add math equation in my blog"
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


Add the following into `extend_head.html`
```html
{{ if or .Params.math .Site.Params.math }}
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.css"
  integrity="sha384-AfEj0r4/OFrOo5t7NnNe46zW/tFgW6x/bCJG8FqQCEo3+Aro6EYUG4+cU+KJWu/X" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.js"
  integrity="sha384-g7c+Jr9ZivxKLnZTDUhnkOnsh30B4H0rpLUpJ4jAIKs4fnJI+sEnkvrMWph2EDg4" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/contrib/auto-render.min.js"
  integrity="sha384-mll67QQFJfxn0IYznZYonOWZ644AWYC+Pt2cHqMaRhXVrursRwvLnLaebdGIlYNa" crossorigin="anonymous" onload="renderMathInElement(document.body, 
//   The following is to parse inline math equation
  {
      delimiters: [
                  {left: '$$', right: '$$', display: true},
                  {left: '\\[', right: '\\]', display: true},
                  {left: '$', right: '$', display: false},
                  {left: '\\(', right: '\\)', display: false}
              ]
          }
);"></script>
{{ end }}
```

Then in the markdown file, in the header section we add `math: true`.

### Methods to Show Math
The the above setup, you can use the following ways in the markdown writeup. 
```html
\\(E=mc^2\\) 
$$E=mc^2$$
$E=mc^2$
```