---
layout: post
title: Windows7解决安装anaconda报错failed to create anacoda menu 
categories: 数据分析
description: 在Windows上安装anaconda时，总是报failed to create anaconda menu的错误。
date: 2017-9-10 21:01:02
tags: [机器学习,数据分析,anaconda]

---
# 1. 问题描述
在Windows7上安装anaconda时，快到安装完成了，总是报failed to create anaconda menu的错误，重试也无法解决问题，然后选择忽略，忽略，提示安装完成，但是没有菜单，也没有快捷方式，只能在控制面板——>程序和功能里找到对应的软件已经安装。

# 2. 解决方案
点击win键+R，输入cmd，回车，进入控制台。
使用"cd"命令进入anaconda的安装位置，我的安装位置为D:\ProgramData\Anaconda2\
然后执行 python .\Lib\\_nsis.py mkmenus
执行过程会出现一连串的successful，表明操作成功。
据体过程如下图所示
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-9-10/11466514.jpg)
</center>

上述命令执行完成之后可在菜单栏看到如下图所示的一系列图标
<center>
![结果图](http://otue1rxl3.bkt.clouddn.com/17-9-10/26560664.jpg)
</center>

至此，该问题完美解决。
【转】[windows安装anaconda 报错failed to create anacoda menu ？](http://blog.csdn.net/lixiangyong123/article/details/55816168)