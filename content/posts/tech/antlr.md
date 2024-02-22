---
title: "ANTLR Parser Generator"
date: 2022-08-08T12:01:14-07:00
lastmod: 2022-08-08T12:01:14-07:00
author: ["Jun"]
keywords: 
- 
categories: # 没有分类界面可以不填写
- 
tags: # 标签
- Parsing
description: "How to use antlr framework to generate code parser"
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
Parsing as the very first step of compiling is important for language analysis. ANTLR is a powerful tool to generate parsers. In this blog, we're trying to understand more about ANTLR and its usage. 

## ANTLR Grammar

(1) ANTLR has two kinds of labels: *alternative labels* and *rule elements labels*; both can be useful. We assume you are familiar with these two kinds of labels, but here it is an example.

```
expression  : left=expression '*' right=expression #multiplication

​           | expression '+' expression            #addition      

​           | NUMBER                               #atom

​           ;
```

Alternative labels are the one that follows an `#`, the rule element labels are the one preceding the = sign. They serve two different purposes. The first ones facilitate act differently for each alternative while the second ones facilitate accessing the different parts of the rules.

For alternative labels, if you label one alternative, you have to label all alternatives, because there will be no base node. Rule element labels instead provides an alternative way to access the content parsed by the sub-rule to which the label is assigned. 

(2) For the following grammar, both `item` nad `clause` will be parsed as a list in the parsing tree. They can be visited using `ctx.items()` and `ctx.clause()`

```````
expression : items* 

​          | clause+
```````

(3) Parsing nested rule sometimes can be very challenging, the solution is we move the nested on into a new rule, or using labels mentioned above. 

## ANTLR Lexer

(1) In a lexer rule, the characters inside square brackets define a character set. So `["]` is the set with the single character `"`. Being a set, every character is either in the set or not, so defining a character twice, as in `[""]` makes no difference, it's the same as `["]`.

`~` negates the set, so `~["]` means *any character except `"`*.

(2) In lexer or grammar, literals are marked out by quote. In the following example, `namedChars` will be single-quote quoted char list and ended with `X` or `x` 

```
namedChars : '\'' Chars '\''[Xx]
```

Note that the grammar doesn't count any spaces in the char.



## ANTLR Parser Visitor and Listener Mode

ANTLR parser provides two kinds of mechanisms to access the parsing nodes. First is listener mode: we can enter a node to perform actions based on our needs. Second is visitor mode: we can visit all parsing tree nodes top-down, left-right sequentially. [This repo](https://github.com/AlanHohn/antlr4-python) provides simple but useful tutorials about how this works. 















## Reference

[1] [This blog](https://tomassetti.me/best-practices-for-antlr-parsers/) is very useful for to me when I wrote this summary doc. 

[2] StackOverflow

[3] The definitive ANTLR Guide book

