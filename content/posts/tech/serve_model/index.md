---
title: "LLM Playground"
date: 2022-02-08T12:01:14-07:00
lastmod: 2022-02-08T12:01:14-07:00
author: ["Jun"]
keywords: 
- 
categories: # 没有分类界面可以不填写
- 
tags: # 标签
- Serving
description: "How to serve your model locally for a demo"
weight:
slug: ""
draft: false # 是否为草稿
comments: true # 本页面是否显示评论
reward: false # 打赏
mermaid: true #是否开启mermaid
showToc: true # 显示目录
TocOpen: true # 自动展开目录
hidemeta: false # 是否隐藏文章的元信息，如发布日期、作者等
disableShare: true # 底部不显示分享栏
showbreadcrumbs: true #顶部显示路径
cover:
    image: "" #图片路径例如：posts/tech/123/123.png
    zoom: # 图片大小，例如填写 50% 表示原图像的一半大小
    caption: "" #图片底部描述
    alt: ""
    relative: false
---
### LLM Serving
Working on LLMs often entails us to conduct a demo for real-time test. Sometimes we have to set things up so that co-worker can play with our model to find out the issues there. 
An eassy way is to use Flask. 
```python
import flask

app = flask.Flask(__name__)

@app.route('/')
def index():
 return "<h3>My LLM Playground</h3>"
```

### Start the Server
Start the server, we can run 
```bash
ApiServicePort=xxxx python3 serve.py
```
### Front-End
If we use flask `render_template` to provide the front end, then we can use the following to ways to launch the app,
```bash
# method 1
flask run

# method 2
python3 app.py
```

Another way is to use `streamlit`. Streamlit is an open-source Python library that allows developers to create web applications for data science and machine learning projects with minimal effort. It is designed to simplify the process of turning data scripts into shareable web apps, enabling users to interact with data and models through a web browser. 
If we use `streamlit`, we can run with
```bash
streamlit run app.py
```
Usually we first star the serve and specific the port to listening on. Then pull up the front end page. 


The page will be like the following, simple and easy!!
<!-- ![](images/playground.png) -->
<p align="center">
    <img alt="gopher dataset" src="images/playground.png" width="90%"/>
    <br>
    <em>LLM Playground</em>
    <br>
</p>


### References
<!-- [1] [LiteLLM](https://github.com/BerriAI/litellm/) <br> -->
[1] [Openplayground](https://github.com/nat/openplayground) <br>
<!-- [2] https://www.youtube.com/watch?v=bluclMxiUkA <br> -->