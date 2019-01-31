---

layout: post
title: Win10 Word2016 64bit 添加 Mathtype支持
categories: others
description: Win10 Word2016 64bit 添加 Mathtype支持
date: 2018-9-27 10:36:21
tags: [Win10, office2016, mathtype]
mathjax: true

---

首先安装office 2016 64bit
然后安装Mathtype6.9 64bit

# 1 首先找到以下路径
【1】**‪E:\Software\MathType\MathPage\32\MathPage.wll**
【2】**E:\Software\MathType\Office Support\64\MathType Commands 6 For Word 2013.dotm**
【3】**E:\Software\MathType\Office Support\64\WordCmds.dot**
【4】**C:\Program Files (x86)\Microsoft Office\root\Office16**
【5】**C:\Program Files (x86)\Microsoft Office\root\Office16\STARTUP**

> # 注意
> 【1】中所示路径为**32**，而不是**64**。至于为什么是32，我也不得而知，请参考[知乎：MathType 与 Office 2016 不兼容怎么办？](https://www.zhihu.com/question/37390635)

# 2 复制粘贴
将【1】复制粘贴到【4】；
将【2】【3】复制粘贴到【5】

# 3 配置完成，亲测有效！
打开Word2016，可以看到菜单栏已经成功加载Mathtype。如下图：

<center>![](http://otue1rxl3.bkt.clouddn.com/18-9-27/73773149.jpg)</center>

> PS: 此处只给出其中一种情况，也有可能存在其他情况，具体百度。
