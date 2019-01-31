---

layout: post
title: NVIDIA Jetson TX2 学习笔记（1）——开箱
categories: Jetson TX2
description: NVIDIA Jetson TX2 开箱
date: 2018-4-4 09:07:20
tags: [Jetson TX2, NVIDIA, 深度学习]

---

　　在此之前对于开发板的接触应该还是蛮多的，从51单片机开发板到STM32开发板，再到Intel的伽利略或者是ARM A9的嵌入式开发板。在第一次亲手接到这高端大气上档次，相当不低调，仍旧抵挡不住其奢华和内涵的NVIDIA Jetson TX2时，顿时觉得以前的眼界实在是太窄了。第一次拿到板子自然是小心翼翼，毕竟这么贵，和以前的比起来都是便宜货了。话不多说，先介绍一波Jetson TX2。

# 1 Jetson TX2 简介

　　Jetson TX2是基于 NVIDIA Pascal™ 架构的 AI 单模块超级计算机，性能强大（1 TFLOPS），外形小巧，节能高效（7.5W），非常适合机器人、无人机、智能摄像机和便携医疗设备等智能终端设备。  
Jatson TX2 与 TX1 相比，内存和 eMMC 提高了一倍，CUDA 架构升级为 Pascal，每瓦性能提高一倍，支持 Jetson TX1 模块的所有功能，支持更大、更深、更复杂的深度神经网络。

<center>![](http://otue1rxl3.bkt.clouddn.com/18-4-25/78955814.jpg)
图1：Jetson TX2内部结果</center>

## 1.1 处理单元

- dual-core NVIDIA Denver2 + quad-core ARM Cortex-A57

- 256-core Pascal GPU

- 8GB LPDDR4, 128-bit interface

- 32GB eMMC

- 4kp60 H.264/H.265 encoder & decoder

- Dual ISPs (Image Signal Processors)

- 1.4 gigapixel/sec MIPI CSI camera ingest

## 1.2 接口&外设

- HDMI 2.0

- 802.11a/b/g/n/ac 2×2 867Mbps WiFi

- Bluetooth 4.1

- USB3, USB2、

- 10/100/1000 BASE-T Ethernet

- 12 lanes MIPI CSI 2.0, 2.5 Gb/sec per lane

- PCIe gen 2.0, 1×4 + 1×1 or 2×1 + 1×2

- SATA, SDcard

- dual CAN bus

- UART, SPI, I2C, I2S, GPIOs

# 2 开箱

　　Jetson TX2开发套件包含了以下组件：

- mini-ITX参考载板

- Jetson TX2模块

- 风扇和散热器（预组装）

- 5MP CSI相机模块（带Omnivision OV5693）

- WiFi/BT天线

- USB OTG适配器

- 19VDC电源适配器

<center>![](http://otue1rxl3.bkt.clouddn.com/18-4-25/75193664.jpg)![](http://otue1rxl3.bkt.clouddn.com/18-4-25/71868286.jpg)
图2：Jetson TX2开发套件</center>

# 3 上电测试

- 将天线按要求安装到Jetson TX2开发板，将显示器与开发板连接（如显示器不支持HDMI，则需要HDMI/VGA转接线），连接好键盘鼠标（需要一个USB集线器），将电源适配器和Jetson TX2开发板正确连接。
<center>![](http://otue1rxl3.bkt.clouddn.com/18-4-25/31235480.jpg)</center>

- 按下POWER按钮，可以观察到显示器滚动显示开机信息，最终显示如下信息。
<center>![](http://otue1rxl3.bkt.clouddn.com/18-4-25/75529726.jpg)</center>

- 按照显示器上显示的信息，按步骤进行操作

   - Step 1）：更改目录到NVIDIA installation目录。</br>cd "${HOME}/NVIDIA-INSTALLER"

   - Step 2）：运行安装脚本，提取和安装NVIDIA Linux驱动。</br>sudo ./installer.sh

   - Step 3）：重启系统，进入Ubuntu桌面环境。</br>sudo reboot

- 重启进入Ubuntu桌面环境。

- 选择用户“nvidia”，输入密码“nvidia”。

<center>![](http://otue1rxl3.bkt.clouddn.com/18-4-25/14163062.jpg)
图3：Jetson TX2 Ubuntu桌面</center>

　　至此，开箱上电测试已经结束。按照套路，接下来就可以进行刷机操作了，将Jetson TX2的Jetpack更新到最新版本。


