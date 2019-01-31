---
layout: post
title: Caffe-Python接口常用API总结
categories: Caffe
description: Caffe Python接口常用API
date: 2018-5-15 21:23:53
tags: [API, caffe]
mathjax: true

---

本文整理了pycaffe中常用的caffe Python API

<!-- TOC -->

- [导入caffe](#导入caffe)
- [1 caffe.NetSpec() 定义一个网络](#1-caffenetspec-定义一个网络)
- [2 Layers定义](#2-layers定义)
    - [2.1 Data Layers 数据层定义](#21-data-layers-数据层定义)
        - [2.1.1 L.Data()](#211-ldata)
        - [2.1.2 L.HDF5Data()](#212-lhdf5data)
        - [2.1.3 L.ImageData()](#213-limagedata)
    - [2.2 Vision Layers 视觉层定义](#22-vision-layers-视觉层定义)
        - [2.2.1 L.Convolution()](#221-lconvolution)
        - [2.2.2 L.Pooling()](#222-lpooling)
        - [2.2.3 L.LRN()](#223-llrn)
        - [2.2.4 L.InnerProduct()](#224-linnerproduct)
    - [2.3 Activation Layers 激活层定义](#23-activation-layers-激活层定义)
        - [2.3.1 L.ReLU()](#231-lrelu)
        - [2.3.2 L.Sigmoid()](#232-lsigmoid)
        - [2.3.3 L.TanH()](#233-ltanh)
        - [2.3.4 L.AbsVal()](#234-labsval)
        - [2.3.5 L.Power()](#235-lpower)
        - [2.3.6 L.BNLL()](#236-lbnll)
    - [2.4 Other Layers 其他常用层定义](#24-other-layers-其他常用层定义)
        - [2.4.1 L.SoftmaxWithLoss()](#241-lsoftmaxwithloss)
        - [2.4.2 L.Softmax()](#242-lsoftmax)
        - [2.4.3 L.Accuracy()](#243-laccuracy)
        - [2.4.4 L.Dropout()](#244-ldropout)

<!-- /TOC -->

# 导入caffe

```python
import caffe
from caffe import layers as L
from caffe import params as P
```

# 1 caffe.NetSpec() 定义一个网络

```python
n = caffe.NetSpec()
```

上述代码是获取Caffe的一个Net，我们只需不断的填充这个n，最后把n输出到文件*.prototxt。

```python
n.to_proto()
```

将定义好的网络输出到*.prototxt文件中。

# 2 Layers定义

## 2.1 Data Layers 数据层定义

### 2.1.1 L.Data()

lmdb/leveldb Data层定义

```python
################################################################################
# data,label:   为top的名称
# Data():       表示该层类型为数据层，数据来自于levelDB或者LMDB。
# source:       lmdb数据目录。
# backend:      选择是采用levelDB还是LMDB，默认是levelDB。
# batch_size:   每一次处理数据的个数。
# ntop:         表明有多少个blobs数据输出，示例中为2，代表着data和label。
# phase:        0:表示TRAIN
#               1:表示TEST
# transform_param:  数据预处理
#   scale:      归一化。1/255即将输入数据从0-255归一化到0-1之间。
#   crop_size:  对图像进行裁剪。如果定义了crop_size，那么在train时会对大
#               于crop_size的图片进行随机裁剪，而在test时只是截取中间部分。
#   mean_value: 图像通道的均值。三个值表示RGB图像中三个通道的均值。
#   mirror:     图像镜像。True为使用镜像。
################################################################################
n.data, n.label = L.Data(
    source=lmdb,
    backend=P.Data.LMDB，
    batch_size=batch_size,
    ntop=2,
    include=dict(phase=0)
    transform_param=dict(
                        scale=1./225,
                        crop_size=227,
                        mean_value=[104, 117, 123],
                        mirror=True
                        )
)
```

其效果如下

```python
layer {
    name: "data"
    type: "Data"
    top: "data"
    top: "label"
    include {
        phase: TRAIN
    }
    transform_param {
        scale: 0.00444
        mirror: true
        crop_size: 227
        mean_value: 104
        mean_value: 117
        mean_value: 123
    }
    data_param {
        source: "path/to/lmdb"
        batch_size: 64
        backend: LMDB
    }
}
```

### 2.1.2 L.HDF5Data()

HDF5 Data层定义

```python
################################################################################
# data,label:   为top的名称。
# HDF5Data():   表示该层类型为HDF5数据层。
# source:       读取文件名称。
# batch_size：  每一次处理数据的数量。
# include:      附加参数。
#   phase:      TRAIN或者TEST。
################################################################################
n.data, n.label = L.HDF5Data(
    hdf5_data_param={
        'source': './training_data_path.txt',
        'batch_size': batch_size
    }
    include={
        'phase': caffe.TRAIN
    }
)
```

其效果如下：

```python
layer {
    name: "data"
    type: "HDF5Data"
    top: "data"
    top: "label"
    hdf5_data_param {
        source: "examples/hdf5_classification/data/train.txt"
        batch_size: 10
    }
    include {
        phase: caffe.TRAIN
    }
}
```

### 2.1.3 L.ImageData()

Image Data层定义

```python
################################################################################
# data,label:   为top的名称
# ImageData():  表示该层类型为数据层，数据来自于图片。
# source:       一个文本文件的名称，每一行给定一个图片文件的名称和标签。
# batch_size:   每一次处理数据的个数。
# new_width:    图片resize的宽。（可选）
# new——height:  图片resize的高。（可选）
# ntop:         表明有多少个blobs数据输出，示例中为2，代表着data和label。
# transform_param:  数据预处理
#   crop_size:  对图像进行裁剪。如果定义了crop_size，那么在train时会对大
#               于crop_size的图片进行随机裁剪，而在test时只是截取中间部分。
#   mean_value: 图像通道的均值。三个值表示RGB图像中三个通道的均值。
#   mirror:     图像镜像。True为使用镜像。
################################################################################
n.data, n.label = L.ImageData(
            source=list_path,
            batch_size=batch_size,
            new_width=48,
            new_height=48,
            ntop=2,
            transform_param=dict(
                                crop_size=40,
                                mean_value=[104, 117, 123],
                                mirror=True
                               )
           )
```

其效果如下：

```python
layer {
    name: "data"
    type: "ImageData"
    top: "data"
    top: "label"
    transform_param {
        mirror: false
        crop_size: 227
        mean_value: 104
        mean_value: 117
        mean_value: 123
    }
    image_data_param {
        source: "examples/_temp/file_list.txt"
        batch_size: 50
        new_height: 256
        new_width: 256
    }
}
```

## 2.2 Vision Layers 视觉层定义

视觉层包括Convolution, Pooling, Local Response Normalization (LRN)、Fully Connection等层。

### 2.2.1 L.Convolution()

```python
################################################################################
# bottom:       上一层数据输出。
# kernel_size:  卷积核大小。
# stride:       卷积核的步长，如果卷积核的长和宽不等，需要使用kernel_h和kernel_w
#               分别设定。
# num_output:   卷积核的数量。
# pad:          扩充边缘，默认为0，不扩充。扩充的时候上下、左右对称，比如卷积核为5*5，
#               那么pad设置为2，则在四个边缘都扩充两个像素，即宽和高都扩充4个像素，这
#               样卷积运算之后特征图不会变小。也可以使用pad_h和pad_w来分别设定。
# group:        分组，默认为1组。如果大于1，我们限制卷积的连接操作在一个子集内，如果
#               我们根据图像的通道来分组，那么第i个输出分组只能与第i个输入分组进行连接。
# weight_filler:权值初始化方式。默认为“constant”，值全为0，很多时候使用“xavier”算法
#               进行初始化。可选方式有：
#               constant:           常数初始化（默认为0）
#               gaussian:           高斯分布初试化权值
#               positive_unitball:  该方式可防止权值过大
#               uniform:            均匀分布初始化
#               xavier:             xavier算法初始化
#               msra:
#               billinear:          双线性插值初始化
# bias_filler:  偏置项初始化。一般设置为“constant”，值全为0.
# bias_term:    是否开启偏置项，默认为true，开启。
################################################################################
n.conv1 = L.Convolution(
                bottom,
                kernel_size=ks,
                stride=stride,
                num_output=nout,
                pad=pad,
                group=group,
                weight_filler=dict(type='xavier'),
                bias_filler=dict(type='constant'),
                bias_term=True
             )
```

其效果如下：

```python
layer {
    name: "conv1"
    type: "Convolution"
    bottom: "data"
    top: "conv1"
    param {
        lr_mult: 1
    }
    param {
        lr_mult: 2
    }
    convolution_param {
        num_output: 20
        kernel_size: 5
        stride: 1
        pad:2
        weight_filler {
            type: "xavier"
        }
        bias_filler {
            type: "constant"
        }
    }
}
```

### 2.2.2 L.Pooling()

池化层pool定义

```python
################################################################################
# bottom:       上一层数据输出。
# pool:         池化方式，默认为MAX。目前可用的方法有MAX, AVE, 或STOCHASTIC。
# kernel_size:  池化的核大小。也可以用kernel_h和kernel_w分别设定。
# stride:       池化的步长，默认为1。一般我们设置为2，即不重叠。也可以用stride_h和
#               stride_w来设置。
################################################################################
n.pool1 = L.Pooling(
            bottom,
            pool=P.Pooling.MAX,
            kernel_size=ks,
            stride=stride
         )
```

其效果如下：

```python
layer {
    name: "pool1"
    type: "Pooling"
    bottom: "conv1"
    top: "pool1"
    pooling_param {
        pool: MAX
        kernel_size: 3
        stride: 2
    }
}
```

### 2.2.3 L.LRN()

Local Response Normalization (LRN)层,此层是对一个输入的局部区域进行归一化，达到“侧抑制”的效果。可去搜索AlexNet或GoogLenet，里面就用到了这个功能.

```python
################################################################################
# bottom:       上一层数据输出。
# local_size:   默认为5。如果是跨通道LRN，则表示求和的通道数；如果是在通道内LRN，
#               则表示求和的正方形区域长度。
# alpha:        默认为1，归一化公式中的参数。
# beta:         默认为5，归一化公式中的参数。
################################################################################
n.norm1 = L.LRN(
        bottom,
        local_size=5,
        alpha=1e-4,
        beta=0.75
        )
```

其效果如下：

```python
layers {
    name: "norm1"
    type: LRN
    bottom: "pool1"
    top: "norm1"
    lrn_param {
        local_size: 5
        alpha: 0.0001
        beta: 0.75
    }
}
```

### 2.2.4 L.InnerProduct()

全连接层，把输入当作成一个向量，输出也是一个简单向量（把输入数据blobs的width和height全变为1）。

```python
n.fc = L.InnerProduct(
                bottom,
                num_output=nout,
                weight_filler=dict(type='xavier'),
                bias_filler=dict(type='constant'),
                bais_term=True
              )
```

其效果如下：

```python
layer {
    name: "fc"
    type: "InnerProduct"
    bottom: "pool2"
    top: "fc"
    inner_product_param {
        num_output: 500
        weight_filler {
            type: "xavier"
        }
        bias_filler {
            type: "constant"
        }
    }
}
```

## 2.3 Activation Layers 激活层定义

### 2.3.1 L.ReLU()

ReLU是目前使用最多的激活函数，主要因为其收敛更快，并且能保持同样效果。
标准的ReLU函数为max(x, 0)，当x>0时，输出x; 当x<=0时，输出0。

```python
################################################################################
# negative_slope：  默认为0. 对标准的ReLU函数进行变化，如果设置了这个值，那么数据为
#                   负数时，就不再设置为0，而是用原始数据乘以negative_slope。
################################################################################
n.relu = L.ReLU(
                bottom,
                in_place=True
               )
```

其效果如下：

```python
layer {
    name: "relu1"
    type: "ReLU"
    bottom: "conv1"
    top: "conv1"
}
```

### 2.3.2 L.Sigmoid()

对每个输入数据，利用sigmoid函数执行操作。这种层设置比较简单，没有额外的参数。

```python
################################################################################
# 这种层比较简单，没有额外的参数。
# bottom:       数据输入
################################################################################
n.sigmoid = L.Sigmiod(
                        bottom
                     )
```

其效果如下：

```python
layer {
    name: 'sigmoid'
    type: 'Sigmoid'
    bottom: 'in'
    top: 'sigmoid'
}
```

### 2.3.3 L.TanH()

利用双曲正切函数对数据进行变换。

```python
################################################################################
# 这种层比较简单，没有额外的参数。
# bottom:       数据输入
################################################################################
n.tanh = L.TanH(
                bottom
               )
```

其效果如下：

```python
layer {
    name: 'tanh'
    type: 'TanH'
    bottom: 'in'
    top: 'tanh'
}
```

### 2.3.4 L.AbsVal()

求每个输入数据的绝对值。

```python
################################################################################
# 这种层比较简单，没有额外的参数。
# bottom:       数据输入
################################################################################
n.abs = L.AbsVal(
                bottom
                )
```

其效果如下：

```python
layer {
    name: 'abs'
    type: 'AbsVal'
    bottom: 'in'
    top: 'abs'
}
```

### 2.3.5 L.Power()

对每个输入数据进行幂运算。

f(x)= (shift + scale * x) ^ power

```python
################################################################################
# bottom:       数据输入
# power:        （可选）默认为1
# scale:        （可选）默认为1
# shift:        （可选）默认为0
################################################################################
n.power = L.Power(
                  bottom,
                  power=2,
                  scale=1,
                  shift=0
                 )
```

其效果如下：

```python
layer {
    name: "layer"
    bottom: "in"
    top: "out"
    type: "Power"
    power_param {
        power: 2
        scale: 1
        shift: 0
    }
}
```

### 2.3.6 L.BNLL()

binomial normal log likelihood的简称，二项式正态对数似然。

f(x)=log(1 + exp(x))

```python
################################################################################
# 这种层比较简单，没有额外的参数。
# bottom:       数据输入
################################################################################
n.bnll = L.BNLL(
                bottom
               )
```

## 2.4 Other Layers 其他常用层定义

本节讲解一些其它的常用层，包括：softmax_loss层，Inner Product层，accuracy层，reshape层和dropout层及其它们的参数配置。

### 2.4.1 L.SoftmaxWithLoss()

softmax-loss层和softmax层计算大致是相同的。softmax是一个分类器，计算的是类别的概率（Likelihood），是Logistic Regression 的一种推广。Logistic Regression 只能用于二分类，而softmax可以用于多分类。

softmax与softmax-loss的区别：

>softmax计算公式:
>
>$$p_j = \frac {e{_{j} ^{o} }}{\sum _k e{^{o_k} }} $$
>
>softmax-loss计算公式:
>
>$$L = - \sum _{j} y_i \log p_j$$

用户可能最终目的就是得到各个类别的概率似然值，这个时候就只需要一个 Softmax层，而不一定要进softmax-Loss 操作；或者是用户有通过其他什么方式已经得到了某种概率似然值，然后要做最大似然估计，此时则只需要后面的softmax-Loss 而不需要前面的 Softmax 操作。因此提供两个不同的 Layer 结构比只提供一个合在一起的Softmax-Loss Layer 要灵活许多。

不管是softmax layer还是softmax-loss layer,都是没有参数的，只是层类型不同而已。

```python
################################################################################
# 该层没有额外的参数。
# bottom:       数据输入（n.fc）
# bottom:       数据输入（n.label）
################################################################################
n.loss = L.SoftmaxWithLoss(
                            n.fc,
                            n.label
                          )
```

其效果如下：

```python
layer {
    name: "loss"
    type: "SoftmaxWithLoss"
    bottom: "fc"
    bottom: "label"
    top: "loss"
}
```

### 2.4.2 L.Softmax()

```python
################################################################################
# 该层没有额外的参数。
# bottom:       数据输入（n.fc）
################################################################################
n.softmax = L.Softmax(
                      n.fc
                     )
```

其效果如下：

```python
layers {
    bottom: "fc"
    top: "softmax"
    name: "softmax"
    type: "Softmax"
}
```

### 2.4.3 L.Accuracy()

输出分类（预测）精确度，只有test阶段才有，因此需要加入include参数。

```python
################################################################################
# bottom:       数据输入（n.fc）
# bottom:       数据输入（n.label）
# phase:        0:表示TRAIN
#               1:表示TEST
################################################################################
n.accuracy = L.Accuracy(
                    bottom,
                    label，
                    include=dict(phase=1)
                  )
```

其效果如下：

```python
layer {
    name: "accuracy"
    type: "Accuracy"
    bottom: "fc"
    bottom: "label"
    top: "accuracy"
    include {
        phase: TEST
    }
}
```

### 2.4.4 L.Dropout()

Dropout是一个防止过拟合的trick。可以随机让网络某些隐含层节点的权重不工作。

```python
################################################################################
# bottom:       数据输入（n.fc）
# dropout_ratio:
# in_palce:
################################################################################
L.Dropout(
            bottom,
            dropout_ratio=0.5,
            in_place=True
            )
```

其效果如下：

```python
layer {
    name: 'dropout'
    type: 'Dropout'
    bottom: 'fc1'
    top: 'dropout'
    dropout_param {
        dropout_ratio: 0.5
    }
}
```

> 参考文献：
> - [Caffe学习系列(2)：数据层及参数](https://www.cnblogs.com/denny402/p/5070928.html)
> - [Caffe学习系列(3)：视觉层（Vision Layers)及参数](http://www.cnblogs.com/denny402/p/5071126.html)
> - [Caffe学习系列(4)：激活层（Activiation Layers)及参数](http://www.cnblogs.com/denny402/p/5072507.html)
> - [Caffe学习系列(5)：其它常用层及参数](http://www.cnblogs.com/denny402/p/5072746.html)
> - [Caffe-Python接口常用API参考](http://wentaoma.com/2016/08/10/caffe-python-common-api-reference/)