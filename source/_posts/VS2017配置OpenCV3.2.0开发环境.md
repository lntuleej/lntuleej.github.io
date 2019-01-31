---
layout: post
title: Win7下配置VS2017+OpenCV3.2.0
categories: 计算机视觉
description: 本文主要介绍如何在Win7 64位系统上配置VS2017+OpenCV3.2.0开发环境。
date: 2017-10-25 08:58:09
tags: [OpenCV, VS2017]

---

# 1 准备

- 系统：Windows7 64位
- Visial Studio：[Visual Studio Enterprise 2017](https://www.visualstudio.com/zh-hans/downloads/)
- OpenCV：[opencv-3.2.0-vc14.exe](https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.2.0/opencv-3.2.0-vc14.exe/download)
- Cmake：[cmake-3.8.2-win64-x64.msi](https://cmake.org/files/v3.8/cmake-3.8.2-win64-x64.msi)
从官网上下载好以上所列出各软件的安装包。
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-25/1353242.jpg)
图1 准备软件安装包
</center>

# 2 安装各软件
## 2.1 安装Visual Studio Enterprise 2017
关于VS2017的版本介绍以及新增功能的说明在这就不作介绍了，有兴趣可以浏览[Microsoft官网](https://www.visualstudio.com/zh-hans/)查看相关说明。接下来直奔主题，在本文中所采用的安装方式为在线安装，若需要离线安装，请自行百度相关安装教程。

- 双击打开安装包
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-25/48758386.jpg)
</center>

- 单击“继续”

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-25/4495779.jpg)
</center>

VS2017功能非常的强大，几乎支持所有常见语言的开发，但若完整安装VS2017将会占用大量的磁盘空间。由于我仅需要C++的开发环境，因此在本步骤中，我只选中了C++的桌面开发环境，如上图所示。

- 修改安装路径
对于C盘空间充裕的机器来说，可以直接选择默认安装路径，但是一般不建议这么做，但是VS2017比较流氓，那就是不管你选择安装在什么位置，它都会把一些依赖库等安装在C盘，并且占用相当大的空间。在这里我将默认安装位置修改为D盘。如下图所示。
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-25/10718646.jpg)
</center>

- 单击“安装”进行安装
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-25/90347049.jpg)
</center>
由于是在线安装，这个过程时间较长，安装的快慢取决于网速和机器的性能。

- 安装完成
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-25/80815111.jpg)
</center>

安装完成。单击启动，即可启动VS2017。
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-25/30393780.jpg)
VS2017
</center>

至此VS2017的安装基本完成，以后若需要用到其他开发环境，可从菜单栏单击
<center>![](http://otue1rxl3.bkt.clouddn.com/17-10-25/19974973.jpg)</center>

打开VS2017的安装器，然后单击修改，即可选择安装新的开发环境或者卸载已安装开发环境。

## 2.2 安装Cmake3.8.2
- 双击打开安装包，单击“Next”，进入下一步。
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-25/77743980.jpg)
</center>

- 接受协议，单击“Next”，进入下一步。
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-25/29695262.jpg)
</center>

- 将Cmake添加到系统环境变量，并设置生成桌面快捷方式，单击“Next”，进入下一步。
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-25/96957385.jpg)
</center>

- 修改默认安装路径，默认安装在C盘，此处修改为D盘。单击“Next”，进入下一步。
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-25/78092812.jpg)
</center>

- 单击 “install”，进行安装。
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-25/22327765.jpg)
</center>

- 安装中，等待安装完成。
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-25/91272776.jpg)
</center>

- 安装完成，单击“Finish”，结束安装。
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-25/73843283.jpg)
</center>

至此，Cmake3.8.4已经安装完成，双击桌面图标即可打开Cmake。其界面如下图所示。
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-25/35716800.jpg)
</center>

## 2.3 解压OpenCV-3.2.0-vc14.exe
- 双击打开安装包，修改解压位置，此处修改为D盘，单击“Extract”，进入下一步。
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-25/40877630.jpg)
</center>

- 解压过程，待解压完成，该界面会自动关闭。
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-25/84518098.jpg)
</center>

- 解压完成，可在D盘（解压位置）找到opencv文件夹，其结果如图所示。
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-25/80428045.jpg)
</center>

# 3 配置OpenCV
解压后的opencv文件夹中有一个build文件夹，这里边所包含的文件为已编译好的OpenCV库，但是在这已经编译好的库中，我们可以看到\opencv\build\x64\vc14\bin\文件夹下只有少量的动态链接库，因此需要对opencv源文件进行编译，使其生成其他以后可能会用到的动态链接库，避免以后需要使用时带来不必要的麻烦。
## 3.1 编译OpenCV
由于在使用Cmake配置OpenCV编译工程的过程中需要下载如下两个包，
```
opencv_ffmpeg.dll
ippicv_windows_20151201.zip
```
但是由于GWF的原因，这两个包的链接被墙了，因此，在编译OpenCV3.2.0之前，我们需要手动下载这两个包，并复制到对应的文件夹下。
下载链接：
[opencv_ffmpeg.dll](http://download.csdn.net/download/u014291571/10038694)
[ippicv_windows_20151201.zip](http://download.csdn.net/download/u014291571/10038686)
下载完这个两个包后会得到两个压缩包，分别为
```
ffmpeg.zip
ippicv.zip
```
分别解压这两个压缩包。

- 将ffmpeg文件夹中的三个文件复制粘贴到\opencv\sources\3rdparty\ffmpeg\文件夹下，如下图所示：</br><center>![](http://otue1rxl3.bkt.clouddn.com/17-10-25/25929466.jpg)</center>

- 将ippicv文件夹下的downloads文件夹和downloader.cmake复制粘贴到\opencv\sources\3rdparty\ippicv\文件夹下，如下图所示：</br><center>![](http://otue1rxl3.bkt.clouddn.com/17-10-25/28911474.jpg)</center>

接下来进入关键步骤，**使用Cmake配置OpenCV编译工程**。
- 双击Cmake的快捷方式，打开Cmake，并按照下所示对Cmake进行相关设置。</br> (1) Where is the source code：**D:/opencv/sources**</br>(2) Where to build the binaries：**D:/opencv/vc15**</br>上述设置中(1)为opencv源文件目录，这个是固定的，(2)为编译工程目录，这个可以设置为任意位置，但建议按照上述设置进行，方便和后述各步骤中的相关配置进行统一。
- 配置好源目录和工程目录后，单击“Configure”，选择“Visual Studio 15 2017 Win64”，然后单击“Finish”，完成设置，并开始工程配置。具体设置过程如下图所示：</br><center>![](http://otue1rxl3.bkt.clouddn.com/17-10-25/88268072.jpg)</center>
- Cmake工程配置过程</br><center>![](http://otue1rxl3.bkt.clouddn.com/17-10-25/42069070.jpg)</br>Cmake工程配置过程</center>
- Cmake工程配置完成，出现“Configuring done”字样，说明配置成功。如下图所示：</br><center>![](http://otue1rxl3.bkt.clouddn.com/17-10-25/74536136.jpg)</br>Cmake配置成功</center>
- 将上图中红色框中的“BUILD_opencv_world”选中，然后单击“Generate”，生成工程。</br><center>![](http://otue1rxl3.bkt.clouddn.com/17-10-25/78997445.jpg)</br>选中 BUILD_opencv_world</center>
- 生成工程完成，出现“Generating done”，说明生成工程成功。</br><center>![](http://otue1rxl3.bkt.clouddn.com/17-10-25/86181467.jpg)</center>
- 单击“Open Project”打开工程，将下图中圈出位置分别配置为Debug|x64和Release|x64，然后选择“Build——>Buile Solution”在两种配置环境下编译两次。</br><center>![](http://otue1rxl3.bkt.clouddn.com/17-10-25/46426016.jpg)</br>分别配置为Debug|x64和Release|x64</center>
- 分别编译完成后，选择“INSTALL”，然后右键“Build”，分别在Debug|x64和Release|x64下进行。</br><center>![](http://otue1rxl3.bkt.clouddn.com/17-10-25/69818226.jpg)</center>
至此OpenCV的编译工作基本完成。在\opencv\vc15\文件夹下生成了一个install文件夹，这就是最终编译好的OpenCV3.2.0的库。但是，值得注意的是，在这已经编译好的库中，\opencv\vc15\install\bin\文件下没有生成下面这几个库:
```
opencv_core320d.lib
opencv_highgui320d.lib
opencv_imgproc320d.lib
opencv_core320.lib
opencv_highgui320.lib
opencv_imgproc320.lib
```
此时，只需在CMake配置时把BUILD_opencv_world取消选中，在重新按照上述步骤执行一遍即可生成上述动态链接库。一般在以后需要用到哪个库就选中或者取消选中对应的库进行编译即可。
关于vc15文件夹下的各个文件只需保留**install**文件夹即可，其他均可删除。
## 3.2 配置OpenCV的环境变量
- 选中“我的电脑，右键——>属性——>高级系统设置”</br><center>![](http://otue1rxl3.bkt.clouddn.com/17-10-25/90045500.jpg)</center>
- 选择“高级——>环境变量”</br><center>![](http://otue1rxl3.bkt.clouddn.com/17-10-25/91203204.jpg)</center>
- 选择“系统变量——>Path——>编辑”，将D:\opencv\vc15\install\bin添加到变量值后面，用“;(英文)”和前面的路径隔开。单击确定完成配置。</br><center>![](http://otue1rxl3.bkt.clouddn.com/17-10-25/36769976.jpg)</center>

至此，环境变量配置完成，VS2017+OpenCV3.2.0的开发环境基本搭建完成，接下来进行测试。

# 4 测试
- 打开VS2017，“File——>New——>Project”；</br><center>![](http://otue1rxl3.bkt.clouddn.com/17-10-25/63746374.jpg)</center>
- 选择“Installed——>Visual C++——>Windows Desktop——>Windows Console Application”,然后按照如下如所示设置(name和location可任意设置)；</br><center>![](http://otue1rxl3.bkt.clouddn.com/17-10-25/98296129.jpg)</center>
- 在Test_OpenCV.cpp中添加下代码

```C++
// Test_OpenCV.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp> 

using namespace cv;

int main()
{
    // 读入一张图片（图片）    
	Mat img = imread("2.jpg");
	// 创建一个名为 "Pic"窗口    
	namedWindow("Pic");
	// 在窗口中显示图片    
	imshow("Pic", img);
	// 等待6000 ms后窗口自动关闭    
	waitKey(6000);
}
```
- 双击“Property Manager——>Test_OpenCV——>Debug|x64”，选择“VC++ Directories——>Include Directories——>Edit”；将</br>**E:\opencv\vc15\install\include\opencv2;</br>E:\opencv\vc15\install\include\opencv;</br>E:\opencv\vc15\install\include**</br>添加进去（盘符根据实际情况修改），如下图所示：</br><center>![](http://otue1rxl3.bkt.clouddn.com/17-10-25/99692292.jpg)![](http://otue1rxl3.bkt.clouddn.com/17-10-25/68277619.jpg)</center>
- 选择“VC++ Directories——>Library Directories——>Edit”；将</br>**E:\opencv\vc15\install\lib**</br>添加进去，如下图所示：</br><center>![](http://otue1rxl3.bkt.clouddn.com/17-10-25/58692295.jpg)</center>
- 选择“Linker——>Input——>Additional Dependencies——>Edit”，将</br>**opencv_core320d.lib</br>opencv_highgui320d.lib</br>opencv_imgproc320d.lib</br>opencv_world320d.lib**</br>添加进去，如下图所示。</br><center>![](http://otue1rxl3.bkt.clouddn.com/17-10-25/47917699.jpg)</center>
- 双击“Property Manager——>Test_OpenCV——>Release|x64”,按照上述步骤进行相同配置，唯一的区别为，将</br>**opencv_core320d.lib</br>opencv_highgui320d.lib</br>opencv_imgproc320d.lib</br>opencv_world320d.lib**</br>替换为</br>**opencv_core320.lib</br>opencv_highgui320.lib</br>opencv_imgproc320.lib</br>opencv_world320.lib**。
- 按照如下图所示配置，选择“Build——>Build Solution”，然后选择“Debug——>Start Without Debuging”，即可看到运行结果。</br><center>![](http://otue1rxl3.bkt.clouddn.com/17-10-25/99713922.jpg)</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-25/6080550.jpg)
运行结果
</center>

至此，VS2017+OpenCV3.2.0的整个开发环境就全部搭建完成了。