---

layout: post
title: 利用Python对文件及文件夹进行相关操作
categories: Python
description: 利用python对文件夹及文件进行相关操作，即：列出文件夹文件数目、文件夹目录、新建文件夹、移动文件、复制文件等
date: 2018-9-10 19:21:36
tags: [Python]
mathjax: true

---

# 1 统计指定文件夹下所有子文件夹数目及文件数目

```python
# -*- coding: utf-8 -*-
"""
@author: Ad

Operating environment： Windows+python3
"""

# 导入必要的包
import os

root = ".\\"
# 初始化计数器
fileCount = 0
dirCount = 0

# 使用os.walk()递归遍历所有文件夹
# x：目录
# y：子目录
# z：文件
for x, y, z in os.walk(os.path.dirname(root)):
    if len(x) != 0:
        fileCount = 0
        dirCount = 0
        # 列出当前文件夹下的所有子文件夹及文件
        paths = os.listdir(x)
        # 统计当前文件夹下的子文件夹数目及文件数目
        for path in paths:
            # 重构路径，以便于判断当前路径为文件夹还是文件
            path = os.path.sep.join([x, path])
            if os.path.isdir(path):
                dirCount += 1
            elif os.path.isfile(path):
                fileCount += 1

        # 输出结果
        print(x)
        print("\tfiles: {}".format(fileCount))
        print("\tdirs: {}".format(dirCount))
```

# 2 列出目录及子目录下的所有文件并保存为SCV文件

```python
# 初始化文件列表
fileList=[]
# 使用os.walk()递归遍历所有文件夹
for root, dirs, files in os.walk(PATH):
    fileList.append([])
    # 将目录保存到列表的第一列
    fileList[-1].append(root)
    for file in files:
        # 保存当前目录下的文件名到列表
        fileList[-1].append(file)

# 将列表转换为DataFrame
csvFileList = pd.DataFrame(fileList)
# 保存为CSV文件
csvFileList.to_csv("filelist.csv")
```

[完整代码](https://gist.github.com/gitleej/853c6eec9e449ea1087f3287eb4a7098)

# 3 利用python重组文件夹结构

**示例**
> 原始文件夹结构
|——root
|　　|——000
|　　|　　 |——left
|　　|　　 |　　|——index_1.bmp
|　　|　　 |　　|——index_2.bmp
|　　|　　 |　　|——……
|　　|　　 |　　|——middle_1.bmp
|　　|　　 |　　|——middle_2.bmp
|　　|　　 |　　|——……
|　　|　　 |　　|——ring_1.bmp
|　　|　　 |　　|——ring_2.bmp
|　　|　　 |　　|——……
|　　|　　 |——right
|　　|　　 |　　|——index_1.bmp
|　　|　　 |　　|——index_2.bmp
|　　|　　 |　　|——……
|　　|　　 |　　|——middle_1.bmp
|　　|　　 |　　|——middle_2.bmp
|　　|　　 |　　|——……
|　　|　　 |　　|——ring_1.bmp
|　　|　　 |　　|——ring_2.bmp
|　　|　　 |　　|——……
|　　|——001
|　　|——……

将文件夹结构进行重组，重组格式为：
> root\000\left\index_.bmp ——> root\0000
root\000\left\middle_.bmp ——> root\0001
root\000\left\ring_.bmp ——> root\0002
root\000\right\index_.bmp ——> root\0003
root\000\right\middle_.bmp ——> root\0004
root\000\right\ring_.bmp ——> root\0005
…………

## 3.1 导入必要的包

```python
from imutils import paths
import os
import shutil
```

## 3.2 重构文件夹结构

```python
imagePaths = sorted(paths.list_images("./Finger_Vein_Database_oring"))
```

```python
imagePaths[-10:]
```

>['./Finger_Vein_Database_oring/106/right/middle_3.bmp',
 './Finger_Vein_Database_oring/106/right/middle_4.bmp',
 './Finger_Vein_Database_oring/106/right/middle_5.bmp',
 './Finger_Vein_Database_oring/106/right/middle_6.bmp',
 './Finger_Vein_Database_oring/106/right/ring_1.bmp',
 './Finger_Vein_Database_oring/106/right/ring_2.bmp',
 './Finger_Vein_Database_oring/106/right/ring_3.bmp',
 './Finger_Vein_Database_oring/106/right/ring_4.bmp',
 './Finger_Vein_Database_oring/106/right/ring_5.bmp',
 './Finger_Vein_Database_oring/106/right/ring_6.bmp']

```python
total = 0
for imagePath in imagePaths:
    ddir = str(total / 6)
    newdir = os.path.sep.join(["./Finger_Vein_Database", "{}".format(ddir.zfill(4))])
    # 判断文件夹是否存在，如果不存在，则创建文件夹
    if not os.path.exists(newdir):
        print("[INFO] mkdir: {}".format(newdir))
        os.mkdir(newdir)
    print("[INFO] move '{}' to '{}' ".format(imagePath, newdir))
    # 移动文件到指定文件夹
    shutil.move(imagePath,  newdir)
    total += 1
```

> [INFO] move './Finger_Vein_Database_oring/001/left/index_1.bmp' to './Finger_Vein_Database/0000' 
[INFO] move './Finger_Vein_Database_oring/001/left/index_2.bmp' to './Finger_Vein_Database/0000' 
……
[INFO] move './Finger_Vein_Database_oring/001/left/middle_1.bmp' to './Finger_Vein_Database/0001' 
[INFO] move './Finger_Vein_Database_oring/001/left/middle_2.bmp' to './Finger_Vein_Database/0001'
……
[INFO] move './Finger_Vein_Database_oring/001/left/ring_1.bmp' to './Finger_Vein_Database/0002' 
[INFO] move './Finger_Vein_Database_oring/001/left/ring_2.bmp' to './Finger_Vein_Database/0002'
……
……

## 3.3 构建训练集

```python
dataImagePaths = sorted(paths.list_images("./Finger_Vein_Database"))
```

```python
total = 0
for dataImagePath in dataImagePaths:
    ddir = str(total / 6)
    newdir = os.path.sep.join(["./Finger_Vein_Database_train", "{}".format(ddir.zfill(4))])
    # 判断文件夹是否存在，如果不存在，则创建文件夹
    if not os.path.exists(newdir):
        print("[INFO] mkdir: {}".format(newdir))
        os.mkdir(newdir)
    temp = dataImagePath.split(os.path.sep)[-1]
    if("1" in temp) or ("2" in temp) or ("3" in temp) or ("6" in temp):
        print("[INFO] copy '{}' to '{}' ".format(dataImagePath, newdir))
        # 复制文件到指定文件夹
        shutil.copy(dataImagePath,  newdir)
    total += 1
```

> [INFO] mkdir: ./Finger_Vein_Database_train/0000
[INFO] copy './Finger_Vein_Database/0000/index_1.bmp' to './Finger_Vein_Database_train/0000' 
[INFO] copy './Finger_Vein_Database/0000/index_2.bmp' to './Finger_Vein_Database_train/0000' 
[INFO] copy './Finger_Vein_Database/0000/index_3.bmp' to './Finger_Vein_Database_train/0000' 
[INFO] copy './Finger_Vein_Database/0000/index_6.bmp' to './Finger_Vein_Database_train/0000' 
[INFO] mkdir: ./Finger_Vein_Database_train/0001
[INFO] copy './Finger_Vein_Database/0001/middle_1.bmp' to './Finger_Vein_Database_train/0001' 
[INFO] copy './Finger_Vein_Database/0001/middle_2.bmp' to './Finger_Vein_Database_train/0001' 
[INFO] copy './Finger_Vein_Database/0001/middle_3.bmp' to './Finger_Vein_Database_train/0001' 
[INFO] copy './Finger_Vein_Database/0001/middle_6.bmp' to './Finger_Vein_Database_train/0001' 
    ……

## 3.4 构建测试集

```python
total = 0
for dataImagePath in dataImagePaths:
    ddir = str(total / 6)
    newdir = os.path.sep.join(["./Finger_Vein_Database_test", "{}".format(ddir.zfill(4))])
    # 判断文件夹是否存在，如果不存在，则创建文件夹
    if not os.path.exists(newdir):
        print("[INFO] mkdir: {}".format(newdir))
        os.mkdir(newdir)
    temp = dataImagePath.split(os.path.sep)[-1]
    if("4" in temp) or ("5" in temp):
        print("[INFO] copy '{}' to '{}' ".format(dataImagePath, newdir))
        # 复制文件到指定文件夹
        shutil.copy(dataImagePath,  newdir)
    total += 1
```

> [INFO] mkdir: ./Finger_Vein_Database_test/0000
[INFO] copy './Finger_Vein_Database/0000/index_4.bmp' to './Finger_Vein_Database_test/0000'
[INFO] copy './Finger_Vein_Database/0000/index_5.bmp' to './Finger_Vein_Database_test/0000'
[INFO] mkdir: ./Finger_Vein_Database_test/0001
[INFO] copy './Finger_Vein_Database/0001/middle_4.bmp' to './Finger_Vein_Database_test/0001'
[INFO] copy './Finger_Vein_Database/0001/middle_5.bmp' to './Finger_Vein_Database_test/0001'
……

[完整代码](https://gist.github.com/gitleej/c059342c2c6ae34ee5ee575da22c2756)

---

> ### Updata: 加入修改文件夹名称代码
> 当shutil库所处理的**路径中带空格**时，上述代码将会报“NOT FIND”的错误，可以使用下列代码替换或删除原文件夹（文件）名中的空格，以满足shutil的处理要求。
> **代码转自**：[文件及文件夹的重命名-- python实现](https://blog.csdn.net/sinat_31206523/article/details/78824014)

```python
#! /bin/usr/python3
# _*_coding:utf-8_*_

import sys
import os

def cur_file_dir(path):
    # 获取当前文件路径
    # path = sys.path[0]
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        return os.path.dirname(path)

def rename(path):
    # print("当前目录:",path)
    file_list = os.listdir(path)
    # print(file_list)  
    for file in file_list:
        # print(file)         
        old_dir = os.path.join(path,file)          
        filename = os.path.splitext(file)[0]
        # print(filename)
        filetype = os.path.splitext(file)[1]
        # print(filetype)
        old_name = filename + filetype
        print("old name is:", old_name)
        new_filename = filename.replace(" ", "")    # 这里替换的是重点
        new_name = new_filename.replace("", "")     # 如果无法一次替换成功，可以进行多次替换
        print("new name is:", new_name)
        new_dir = os.path.join(path, new_name + filetype)  # 新的文件路径
        os.rename(old_dir, new_dir)                 # 重命名
        # print("DONE")     
        if os.path.isdir(new_dir):
            rename(new_dir)                     # 注意这里是重点，这里使用了递归

if __name__ == "__main__": 
    rename(cur_file_dir("test11"))
    print("ALL DONE!!!")
```