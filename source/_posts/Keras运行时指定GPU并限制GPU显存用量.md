---

layout: post
title: Keras 运行时指定GPU并限制GPU显存用量
categories: Keras
description: Keras在使用GPU训练模型时，默认占用机器所有的GPU及显存，但是，模型在实际运行中并不需要如此多的资源，如果此时有多个模型需要使用GPU跑的话，那么将会受到限制，造成了GPU资源的浪费。因此，为了物尽其用，所以在使用Keras时需要有意识的设置模型运行时所需要使用的GPU以及需要使用的显存大小。
date: 2018-9-10 19:21:36
tags: [Keras, TensorFlow, Python]
mathjax: true

---

# Keras 运行时指定GPU并限制GPU显存用量

　　Keras在使用GPU训练模型时，默认占用机器所有的GPU及显存，但是，模型在实际运行中并不需要如此多的资源，如果此时有多个模型需要使用GPU跑的话，那么将会受到限制，造成了GPU资源的浪费。因此，为了物尽其用，所以在使用Keras时需要有意识的设置模型运行时所需要使用的GPU以及需要使用的显存大小。关于GPU的设置一般分为以下三种情况：

- 1. 指定GPU；
- 2. 限制GPU显存用量；
- 3. 既指定GPU又限制GPU显存用量。

## 自动查看GPU使用情况

```shell
# 1秒刷新一次
$:watch -n 1 nvidia-smi
```

## 1 指定GPU

``` python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
```

这里指定了编号为“2”的GPU，GPU默认是从“0”开始进行编号，大家可以根据实际情况来进行设置。

## 2 限制GPU用量

### 2.1 设置使用GPU的百分比

```python
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

#进行配置，使用30%的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)

# 设置session
KTF.set_session(session )
```

>需要注意的是，虽然代码或配置层面设置了对显存占用百分比阈值，但在实际运行中如果达到了这个阈值，程序有需要的话还是会突破这个阈值。换而言之如果跑在一个大数据集上还是会用到更多的显存。以上的显存限制仅仅为了在跑小数据集时避免对显存的浪费而已。

### 2.2 设置GPU按需使用

```python
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config)

# 设置session
KTF.set_session(sess)
```

## 3 指定GPU并限制GPU用量

这种情况比较简单，就是讲上面两种情况合并使用。

### 3.1 指定GPU+固定GPU用量

```python
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# 指定第二块GPU可用 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3 # 占用30%的显存
sess = tf.Session(config=config)

KTF.set_session(sess)
```

### 3.2 按需使用指定GPU的显存用量

```python
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# 指定第一块GPU可用 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)
```

> # 参考文献
> - [keras指定运行时显卡及限制GPU用量](https://blog.csdn.net/A632189007/article/details/77978058)
> - [keras系列︱keras是如何指定显卡且限制显存用量](https://blog.csdn.net/sinat_26917383/article/details/75633754)