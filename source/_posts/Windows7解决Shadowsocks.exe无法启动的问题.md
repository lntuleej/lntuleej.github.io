---
layout: post
title: Windows7解决Shadowsocks.exe无法启动的问题
categories: Shadowsocks
description: 最近几天在使用Shadowsocks科学上网时，Shadowsocks客服端总是无法启动，于是查找原因，终于解决了。
date: 2017-09-08 15:11:23
tags: [Shadowsocks,Windows7,.NET Framework]
---

# 1. 问题描述

今天在使用Shadowsocks时，双击Shadowsocks客户端图标时，鼠标指针开始转圈，转了半天，什么反应也没有，此时，我已意识到，这玩意完蛋了。

# 2. 分析原因

于是上GitHub找原因，有说是.NET Framework的原因造成的，Shadowsocks 4.0.5 版本需要.NET Framework 4.6.2版本及以上的完整安装才行。

# 3. 解决问题

由于之前Shadowsocks一直都用的好好的，所以，在我的电脑上肯定是有安装.NET Framework 4.6.2 的，但是我也没有卸载过着破玩意啊，为什么么它突然之间就不好使了呢？于是我初步判断应该是.NET Framework损坏。

## 3.1 修复.NET Framework

控制面板——>程序和功能——>Microsoft .NET Framework 4.6.2 右键选择“卸载/更改”

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-9-8/95443098.jpg)

修复页面
</center>

原以为安静的等待其完成即可，结果出现了一些不可描述的未知错误。提示：“安装未成功，安装过程中发生严重错误”。

## 3.2 卸载.NET Framework

修复不成，那就卸载重装。依旧重复上述步骤，而在此过程中选择“从此计算机中删除.NET Framework 4.6.2”，单击下一步。

原以为继续安静等待其完成即可，结果出现了上一步出现的相同的错误。崩溃……

## 3.3 使用Microsoft官方修复工具和卸载工具
先运行NetFxRepairTool.exe修复.NET Framework，若依旧不行。继续运行cleanup_tool.exe卸载对应的.NET Framework，成功之后，开始安装对应的安装包。

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-9-8/33696726.jpg)

NetFxRepairTool.exe运行截图

![](http://otue1rxl3.bkt.clouddn.com/17-9-8/84524184.jpg)

![](http://otue1rxl3.bkt.clouddn.com/17-9-8/63175005.jpg)

![](http://otue1rxl3.bkt.clouddn.com/17-9-8/41287789.jpg)

cleanup_tool.exe 运行截图
</center>



## 3.4 安装.NET Framework 4.6.2

从Microsoft官网下载[.NET Framework 4.6.2](https://www.microsoft.com/en-us/download/confirmation.aspx?id=53344)，进行安装，全部默认即可。

## 3.5 安装更新.NET Framework 4.6.2
原以为重新安装好.NET Frame 4.6.2 之后，我就可以开开心心科学上网了，结果……双击Shadowsocks.exe弹出如下结果

"shadowsocks 非预期错误——无法在DLL"PresentationNative_v0400.dll"中找到名为"IsWindows10RS2OrGreat"的入口点"

其实我也不知道是怎么回事，然后试着重启电脑，结果还是出现相同的情况，接着抱着试一试的态度，安装[Windows更新](http://download.windowsupdate.com/c/msdownload/update/software/secu/2017/04/ndp46-kb4014591-x64_9bcdec650701d5e98aa21b47b50771817c9504df.exe)，完美解决。
