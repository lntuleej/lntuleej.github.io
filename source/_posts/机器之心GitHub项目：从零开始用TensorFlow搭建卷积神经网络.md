---
layout: post
title: 机器之心GitHub项目：从零开始用TensorFlow搭建卷积神经网络【转】
categories: 深度学习
description: 本文的重点是实现，并不会从理论和概念上详细解释深度神经网络、卷积神经网络、最优化方法等基本内容。但是机器之心发过许多详细解释的入门文章或教程，因此，我们希望读者能先了解以下基本概念和理论。当然，本文注重实现，即使对深度学习的基本算法理解不那么深同样还是能实现本文所述的内容。
date: 2017-11-26 21:55:04
tags: [深度学习, tensorflow]

---
# 1.1 张量和图


```python
# 引入 tensorflow
import tensorflow as tf
```


```python
# 定义两个变量
a = tf.constant(2, tf.int16)
b = tf.constant(4, tf.float32)
```


```python
# 定义一张图
graph = tf.Graph()
with graph.as_default():
    # 定义两个变量
    a = tf.Variable(8, tf.float32)
    b = tf.Variable(tf.zeros([2,2]), tf.float32)
```


```python
# 
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    # print(f)
    print(session.run(a))
    print(session.run(b))
```

    8
    [[ 0.  0.]
     [ 0.  0.]]
    


```python
# 声明一个 2 行 3 列的变量矩阵，该变量的值服从标准差为 1 的正态分布，并随机生成
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
```


```python
# 应用变量来定义神经网络中的权重矩阵和偏置项向量：
# truncated_normal()
# Outputs random values from a truncated normal distribution（截断正态分布）.
weights = tf.Variable(tf.truncated_normal([256 * 256, 10]))
biases = tf.Variable(tf.zeros([10]))
print(weights.get_shape().as_list())
print(biases.get_shape().as_list())
```

    [65536, 10]
    [10]
    

# 1.2 占位符和feed_dict

占位符并没有初始值，它只会分配必要的内存。在会话中，占位符可以使用 feed_dict 馈送数据。

feed_dict 是一个字典，在字典中需要给出每一个用到的占位符的取值。在训练神经网络时需要每次提供一个批量的训练样本，如果每次迭代选取的数据要通过常量表示，那么 TensorFlow 的计算图会非常大。因为每增加一个常量，TensorFlow 都会在计算图中增加一个结点。所以说拥有几百万次迭代的神经网络会拥有极其庞大的计算图，而占位符却可以解决这一点，它只会拥有占位符这一个结点。


```python
w1 = tf.Variable(tf.random_normal([1, 2], stddev=1, seed=1))
# 因为需要重复输入x，而每建一个x就会生成一个结点，计算图的效率会低。所以使用占位符
x = tf.placeholder(tf.float32, shape=(1,2))
x1 = tf.constant([0.7, 0.9])

a = x + w1
b = x1 + w1

sees = tf.Session()
sees.run(tf.global_variables_initializer())

# 运行y时将占位符填上，feed_dict为字典，变量名不可变
y_1 = sees.run(a, feed_dict={x:[[0.7, 0.9]]})
y_2 = sees.run(b)
print("使用占位符计算")
print(y_1)
print("常亮计算")
print(y_2)
sees.close
```

    使用占位符计算
    [[-0.11131823  2.38459873]]
    常亮计算
    [[-0.11131823  2.38459873]]
    




    <bound method BaseSession.close of <tensorflow.python.client.session.Session object at 0x000000000CB44780>>



## 下面是使用占位符的案例：


```python
import numpy as np
```


```python
# 计算两点间的欧氏距离
list_of_points1_ = [[1,2], [3,4], [5,6], [7,8]]
list_of_points2_ = [[15,16], [13, 14], [11,12], [9, 10]]

list_of_points1 = np.array([np.array(elem).reshape(1,2) for elem in list_of_points1_])
list_of_points2 = np.array([np.array(elem).reshape(1,2) for elem in list_of_points2_])

# 新建一张图
graph = tf.Graph()

with graph.as_default():
    # 我们使用 tf.placeholder() 创建占位符 ，在 session.run() 过程中再投递数据 
    point1 = tf.placeholder(tf.float32, shape=(1,2))
    point2 = tf.placeholder(tf.float32, shape=(1,2))
    
    # 定义一个函数
    def calculate_eucledian_distance(point1, point2):
        difference = tf.subtract(point1, point2)
        power2 = tf.pow(difference, tf.constant(2.0, shape=(1,2)))
        # 计算一个张量维数的总和
        add = tf.reduce_sum(power2)
        eucledian_distance = tf.sqrt(add)
        return eucledian_distance
    
    dist = calculate_eucledian_distance(point1, point2)
    
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    for ii in range(len(list_of_points1)):
        point1_ = list_of_points1[ii]
        point2_ = list_of_points2[ii]
        
        # 使用feed_dict将数据投入到[dist]中
        feed_dict = {point1: point1_, point2: point2_}
        distance = session.run([dist], feed_dict=feed_dict)
        print("the distance between {} and {} -> {}".format(point1_, point2_, distance))
```

    the distance between [[1 2]] and [[15 16]] -> [19.79899]
    the distance between [[3 4]] and [[13 14]] -> [14.142136]
    the distance between [[5 6]] and [[11 12]] -> [8.485281]
    the distance between [[7 8]] and [[ 9 10]] -> [2.8284271]
    

## 解析一段构建了三层全连接神经网络的代码


```python
import tensorflow as tf
from numpy.random import RandomState
```


```python
# 每次迭代读取的批量为 10
batch_size = 10
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))
```


```python
# None 可以根据batch 大小确定纬度，在shape 的一个纬度上使用None
x = tf.placeholder(tf.float32, shape=(None, 2))
y = tf.placeholder(tf.float32, shape=(None, 1))
```


```python
# 激活函数使用ReLU
# tf.nn.relu() 代表调用 ReLU 激活函数
# tf.matmul() 为矩阵乘法
a = tf.nn.relu(tf.matmul(x,w1))
yhat = tf.nn.relu(tf.matmul(a,w2))
```


```python
# 定义交叉熵为损失函数，训练过程使用Adam算法最小化交叉熵
# tf.clip_by_value(yhat,1e-10,1.0) 这一语句代表的是截断 yhat 的值
cross_entropy = tf.reduce_mean(y * tf.log(tf.clip_by_value(yhat, 1e-10, 1.0)))
# tf.train.AdamOptimizer(learning_rate).minimize(cost_function) 是进行训练的函数，其中我们采用的是 Adam 优化算法更新权重，并且需要提供学习速率和损失函数这两个参数。
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)
data_size = 512
```


```python
# 生成两个特征，共data_size个样本
X = rdm.rand(data_size, 2)
# 定义规则给出样本标签，所用x1+x2<1的样本为正样本，其他为负样本，Y,1为正样本
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]
```


```python
# 创建一个会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w1))
    print(sess.run(w2))
    steps = 11000
    for i in range(steps):
        # 选取每一个批量读取的首位位置，确保在1个epoch内采样训练
        start = i * batch_size % data_size
        end = min(start + batch_size, data_size)
        sess.run(train_step, feed_dict={x:X[start:end], y:Y[start:end]})
        if i % 1000 == 0:
            training_loss = sess.run(cross_entropy, feed_dict={x:X, y:Y})
            print("在迭代 %d 次后，训练损失为 %g"%(i, training_loss))
```

    [[-0.81131822  1.48459876  0.06532937]
     [-2.4427042   0.0992484   0.59122431]]
    [[-0.81131822]
     [ 1.48459876]
     [ 0.06532937]]
    在迭代 0 次后，训练损失为 -0.311019
    在迭代 1000 次后，训练损失为 -11.1082
    在迭代 2000 次后，训练损失为 -11.1082
    在迭代 3000 次后，训练损失为 -11.1082
    在迭代 4000 次后，训练损失为 -11.1082
    在迭代 5000 次后，训练损失为 -11.1082
    在迭代 6000 次后，训练损失为 -11.1082
    在迭代 7000 次后，训练损失为 -11.1082
    在迭代 8000 次后，训练损失为 -11.1082
    在迭代 9000 次后，训练损失为 -11.1082
    在迭代 10000 次后，训练损失为 -11.1082
    

上面的代码定义了一个简单的三层全连接网络（输入层、隐藏层和输出层分别为 2、3 和 1 个神经元），隐藏层和输出层的激活函数使用的是 ReLU 函数。该模型训练的样本总数为 512，每次迭代读取的批量为 10。这个简单的全连接网络以交叉熵为损失函数，并使用 Adam 优化算法进行权重更新。

# tensorflow中的神经网络
## 2.1 简介

![](http://otue1rxl3.bkt.clouddn.com/17-11-27/98620237.jpg)

- 1 定义一些函数，便于对数据进行预处理

## 2.2 导入数据集


```python
import numpy as np
```


```python
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffed_dataset = dataset[permutation, :, :]
    shuffed_labels = labels[permutation]
    return shuffed_dataset, shuffed_labels

def one_hot_encode(np_array):
    return (np.arange(10) == np_array[:, None]).astype(np.float32)

def reformat_data(dataset, labels, image_width, image_height, image_depth):
    np_dataset_ = np.array([np.array(image_data).reshape(image_width, image_height, image_depth) for image_data in dataset])
    np_labels_ = one_hot_encode(np.array(labels,dtype=np.float32))
    np_dataset, np_labels = randomize(np_dataset_, np_labels_)
    return np_dataset, np_labels

def flatten_tf_array(array):
    shape = array.get_shape.as_list()
    return tf.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
```

- 导入数据集


```python
%pwd
```




    'E:\\jupyter notebook'




```python
################################### 导入 mnist 数据集 ################################
from mnist import MNIST
```


```python
mnist_folder = '.\data\mnist'
mnist_image_width = 28
mnist_image_height = 28
mnist_image_depth = 1
mnist_num_labels = 10
mndata = MNIST(mnist_folder)
```


```python
# 导入训练集和测试集
mnist_train_dataset_, mnist_train_labels_ = mndata.load_training()
mnist_test_dataset_, mnist_test_labels_ = mndata.load_testing()
```


```python
# 重新改变数据集的形状
mnist_train_dataset, mnist_train_labels = reformat_data(mnist_train_dataset_, mnist_train_labels_, mnist_image_width, mnist_image_height, mnist_image_depth)
mnist_test_dataset, mnist_test_labels = reformat_data(mnist_test_dataset_, mnist_test_labels_, mnist_image_width, mnist_image_height, mnist_image_depth)
```


```python
# 打印所导入数据集的相关信息
print("There are {} images, each of size {}.".format(len(mnist_train_dataset), len(mnist_train_dataset[0])))
print("Meaning each image has the size of 28*28*1 = {}.".format(mnist_image_width * mnist_image_height * 1))
print("The training set contains the following {} labels: {}.".format(len(np.unique(mnist_train_labels_)), np.unique(mnist_train_labels_)))

print("Training set shape ", mnist_train_dataset.shape, mnist_train_labels.shape)
print("Test set shape ", mnist_test_dataset.shape, mnist_test_labels.shape)
```

    There are 60000 images, each of size 28.
    Meaning each image has the size of 28*28*1 = 784.
    The training set contains the following 10 labels: [0 1 2 3 4 5 6 7 8 9].
    Training set shape  (60000, 28, 28, 1) (60000, 10)
    Test set shape  (10000, 28, 28, 1) (10000, 10)
    


```python
from PIL import Image
import matplotlib.pyplot as plt
```


```python
# 显示图像
im = mnist_train_dataset[0].reshape([28,28])
image = Image.fromarray(im)
plt.imshow(image)
plt.show()
```


![](http://otue1rxl3.bkt.clouddn.com/17-11-27/10952232.jpg)



```python
train_dataset_mnist, train_labels_mnist = mnist_train_dataset, mnist_train_labels
test_dataset_mnist, test_labels_mnist = mnist_test_dataset, mnist_test_labels
```


```python
################################## 导入 CIFAR-10 数据集 ###############################
import pickle

cifar10_folder = '.\\data\\cifar10\\'
train_datasets = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5',]
test_dataset = ['test_batch']
c10_image_height = 32
c10_image_width = 32
c10_image_depth = 3
c10_num_labels = 10
```


```python
# 导入测试集
with open(cifar10_folder + test_dataset[0], 'rb') as f0:
    c10_test_dict = pickle.load(f0, encoding='bytes')
```


```python
c10_test_dataset, c10_test_labels = c10_test_dict[b'data'], c10_test_dict[b'labels']
test_dataset_cifar10, test_labels_cifar10 = reformat_data(c10_test_dataset, c10_test_labels, c10_image_width, c10_image_height, c10_image_depth)
```


```python
# 导入训练集
c10_train_dataset, c10_train_labels = [], []
for train_dataset in train_datasets:
    with open(cifar10_folder + train_dataset, 'rb') as f0:
        c10_train_dict = pickle.load(f0, encoding='bytes')
        c10_train_dataset_, c10_train_labels_ = c10_train_dict[b'data'], c10_train_dict[b'labels']
        
        c10_train_dataset.append(c10_train_dataset_)
        c10_train_labels += c10_train_labels_
```


```python
c10_train_dataset = np.concatenate(c10_train_dataset, axis=0)
train_dataset_cifar10, train_labels_cifar10 = reformat_data(c10_train_dataset, c10_train_labels, c10_image_width, c10_image_height, c10_image_depth)
```


```python
# 打印所导入数据集的相关信息
print("The training set contains the following labels: {}".format(np.unique(c10_train_dict[b'labels'])))
print("Training set shape ", train_dataset_cifar10.shape, train_labels_cifar10.shape)
print("Test set shape ", test_dataset_cifar10.shape, test_labels_cifar10.shape)
```

    The training set contains the following labels: [0 1 2 3 4 5 6 7 8 9]
    Training set shape  (50000, 32, 32, 3) (50000, 10)
    Test set shape  (10000, 32, 32, 3) (10000, 10)
    


```python
c10_im = train_dataset_cifar10[0]
c10_image_r = Image.fromarray(c10_im[:, :, 0]).convert('L')
c10_image_g = Image.fromarray(c10_im[:, :, 1]).convert('L')
c10_image_b = Image.fromarray(c10_im[:, :, 2]).convert('L')
c10_image = Image.merge("RGB", (c10_image_r, c10_image_g, c10_image_b))
plt.imshow(c10_image)
plt.show()
```


![](http://otue1rxl3.bkt.clouddn.com/17-11-27/91538923.jpg)



```python
c10_im[:,:,0]
```




    array([[196, 203, 217, ..., 148, 120,  86],
           [113, 113, 126, ...,  99, 146, 144],
           [127, 116, 159, ..., 190, 203, 174],
           ..., 
           [158, 112,  81, ..., 113,  79,  39],
           [143, 131,  67, ...,  44,  38,  51],
           [123, 125, 111, ...,  53,  79, 120]], dtype=uint8)



## 2.3 搭建一个简单的神经网络

最简单的神经网络为一层线性全连接神经网络，数学上它由矩阵乘法组成。

在学习TensorFlow的过程中，最好先从这样最简单的神经网络开始，然后再接触更加复杂的神经网络。当我们开始研究更复杂的神经网络时，只有图的模型（步骤二）和权重（步骤三）将会发生改变，而其他部分都保持不变。

下面开始构建一个类似的简单神经网络


```python
image_width = mnist_image_width
image_height = mnist_image_height
image_depth = mnist_image_depth
num_labels = mnist_num_labels
```


```python
# 数据集
train_dataset = mnist_train_dataset
train_labels = mnist_train_labels
test_dataset = mnist_test_dataset
test_labels = mnist_test_labels
```


```python
# 迭代次数和学习速率
num_steps = 10001
display_step = 1000
batch_size = 1
learning_rate = 0.001
```


```python
tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, image_depth))
# shape = array.get_shape.as_list()
shape = tf_train_dataset.shape.as_list()
print(shape)
tf_train_dataset_reshape = tf.reshape(tf_train_dataset, [shape[0], shape[1] * shape[2] * shape[3]])
print(tf_train_dataset_reshape.shape)
```

    [1, 28, 28, 1]
    (1, 784)
    


```python
# 构建TensorFlow图
graph = tf.Graph()
with graph.as_default():
    # 1) 首先将数据按照TensorFlow友好的格式输入
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, image_depth))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset, tf.float32)
    
    # 2) 初始化权重矩阵和偏差向量
    # 默认使用tf.truncated_normal() 产生权重矩阵，
    # 默认使用tf.zeros() 产生偏差向量
    weights = tf.Variable(tf.truncated_normal([image_width * image_height * image_depth, num_labels]), tf.float32)
    bias = tf.Variable(tf.zeros([num_labels]), tf.float32)
    
    # 3) 定义模型
    # 由矩阵乘法构成的一层全连接神经网络
    def model(data, weights, bias):
        shape = data.shape.as_list()
        data = tf.reshape(data, [shape[0], shape[1] * shape[2] * shape[3]])
        return tf.matmul(data, weights) + bias
    
    logits = model(tf_train_dataset, weights, bias)
    
    # 4) 计算loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    
    # 5) 选择优化器，进行优化
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    # 6) 训练数据集和测试数据集中图像的预测值分配给变量train_prediction和test_prediction。
    # It is only necessary if you want to know the accuracy by comparing it with the actual values.
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset, weights, bias))
    
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized with learning_rate', learning_rate)
    for step in range(num_steps):
        
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset: (offset + batch_size), :, :, :]
        batch_labels = train_labels[offset: (offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        
        if step % display_step == 0:
            train_accuracy = accuracy(predictions, batch_labels)
            test_accuracy = accuracy(test_prediction.eval(), test_labels)
            message = "step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on test set {:02.2f} %".format(step, l, train_accuracy, test_accuracy)
            print(message)
```

    Initialized with learning_rate 0.001
    step 0000 : loss is 000.00, accuracy on training set 100.00 %, accuracy on test set 4.30 %
    step 1000 : loss is 000.00, accuracy on training set 100.00 %, accuracy on test set 78.88 %
    step 2000 : loss is 10744.32, accuracy on training set 0.00 %, accuracy on test set 78.16 %
    step 3000 : loss is 000.00, accuracy on training set 100.00 %, accuracy on test set 83.81 %
    step 4000 : loss is 000.00, accuracy on training set 100.00 %, accuracy on test set 82.82 %
    step 5000 : loss is 000.00, accuracy on training set 100.00 %, accuracy on test set 85.01 %
    step 6000 : loss is 000.00, accuracy on training set 100.00 %, accuracy on test set 84.43 %
    step 7000 : loss is 000.00, accuracy on training set 100.00 %, accuracy on test set 83.90 %
    step 8000 : loss is 000.00, accuracy on training set 100.00 %, accuracy on test set 84.62 %
    step 9000 : loss is 000.00, accuracy on training set 100.00 %, accuracy on test set 85.80 %
    step 10000 : loss is 000.00, accuracy on training set 100.00 %, accuracy on test set 84.31 %
    

- 上面是原作者的代码，好像有什么问题，目前还太菜了，等懂得多了再来看看；
- 下面的代码源于机器之心，亲测好使；
- 下面我们实现的神经网络共有三层，输入层有 784 个神经元，隐藏层与输出层分别有 500 和 10 个神经元。这所以这样设计是因为 MNIST 的像素为28×28=784，所以每一个输入神经元对应于一个灰度像素点。机器之心执行该模型得到的效果非常好，该模型在批量大小为 100，并使用学习率衰减的情况下迭代 10000 步能得到 98.34% 的测试集准确度，以下是该模型代码：


```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#加载MNIST数据集
mnist = input_data.read_data_sets("./data/mnist/", one_hot=True)
```

    Extracting ./data/mnist/train-images-idx3-ubyte.gz
    Extracting ./data/mnist/train-labels-idx1-ubyte.gz
    Extracting ./data/mnist/t10k-images-idx3-ubyte.gz
    Extracting ./data/mnist/t10k-labels-idx1-ubyte.gz
    


```python
INPUT_NODE = 784     
OUTPUT_NODE = 10     
LAYER1_NODE = 500         
BATCH_SIZE = 100       

# 模型相关的参数
LEARNING_RATE_BASE = 0.8      
LEARNING_RATE_DECAY = 0.99    
REGULARAZTION_RATE = 0.0001   
TRAINING_STEPS = 10000        
MOVING_AVERAGE_DECAY = 0.99 

def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 使用滑动平均类
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2

    else:

        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)  


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 生成隐藏层的参数。
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输出层的参数。
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算不含滑动平均类的前向传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义训练轮数及相关的滑动平均类 
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 定义交叉熵损失函数加上正则项为模型损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularaztion

    # 设置指数衰减的学习率。
    # 函数返回衰减学习速率
    '''
    exponential_decay(
        learning_rate,
        global_step,
        decay_steps,
        decay_rate,
        staircase=False,
        name=None
    )
    '''
    # decayed_learning_rate = learning_rate *decay_rate ^ (global_step / decay_steps)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # 随机梯度下降优化器优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算准确度
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels} 

        # 循环地训练神经网络。
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))

            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" %(TRAINING_STEPS, test_acc)))
```


```python
train(mnist)
```

    After 0 training step(s), validation accuracy using average model is 0.0882 
    After 1000 training step(s), validation accuracy using average model is 0.9766 
    After 2000 training step(s), validation accuracy using average model is 0.9802 
    After 3000 training step(s), validation accuracy using average model is 0.9834 
    After 4000 training step(s), validation accuracy using average model is 0.9844 
    After 5000 training step(s), validation accuracy using average model is 0.9848 
    After 6000 training step(s), validation accuracy using average model is 0.9844 
    After 7000 training step(s), validation accuracy using average model is 0.9856 
    After 8000 training step(s), validation accuracy using average model is 0.9848 
    After 9000 training step(s), validation accuracy using average model is 0.985 
    After 10000 training step(s), test accuracy using average model is 0.9838
    


```python
mnist.train.num_examples
mnist.validation.images.shape
```




    (5000, 784)



## 2.4 多面的TensorFlow
TensorFlow包含很多不同的函数，这意味着相同的操作可以用不同的函数来完成，举个例子：

logits = tf.matmul(tf_train_dataset, weights) + biases

可以替代为：

logits = tf.nn.xw_plus_b(train_dataset, weights, biases)

使用一个高度集成的函数可以使得构建一个包含多层的神经网络变得简单，这个在TensorFlow的API中有很好的体现。例如：conv_2d()或者fully_connected()函数分别构建了卷积层和全连接层，通过这些函数，层级的数量、滤波器的大小/深度、激活函数的类型等都可以明确地作为一个参数。权重矩阵和偏置向量能自动创建，附加激活函数和 dropout 正则化层同样也能轻松构建。

举例：利用层API定义卷积层网络


```python
import tensorflow as tf
w1 = tf.Variable(tf.truncated_normal([filter_size, filter_size, image_depth, filter_depth], stddev=1))
b1 = tf.Variable(tf.zeros([filter_depth]))

layer1_conv = tf.nn.conv2d(data, w1, [1,1,1,1], padding='SAME')
layer1_relu = tf.nn.relu(layer1_conv + b1)
layer1_pool = tf.nn.max_pool(layer1_pool, [1,2,2,1], [1,2,2,1], padding='SAME')
```

以上的代码可以替换为


```python
from tflearn.layers.conv import conv_2d, max_pool_2d
 
layer1_conv = conv_2d(data, filter_depth, filter_size, activation='relu')
layer1_pool = max_pool_2d(layer1_conv_relu, 2, strides=2)
```

在替换上述代码之前需要确认是否安装TensorFlow高级API——[TFlearn](https://github.com/tflearn/tflearn),TFlearn[官网](http://tflearn.org/installation/)。

正如我们所见到的，我们不需要定义权重，偏差或者激活函数，尤其是在需要搭建多层神经网络时，这样可以保持代码紧凑干净。

然而，如果我们是刚开始使用TensorFlow，想要学习搭建不同的神经网络，那么使用高级API就不那么合适，因为，这些高级API把所有事情都做完了。

## 2.5 搭建LeNet5卷积神经网络
LeNet5 卷积网络架构最早是 Yann LeCun 提出来的，它是早期的一种卷积神经网络，并且可以用来识别手写数字。虽然它在 MNIST 数据集上执行地非常好，但在其它高分辨率和大数据集上性能有所降低。对于这些大数据集，像 AlexNet、VGGNet 或 ResNet 那样的深度卷积网络才执行地十分优秀。

因为 LeNet5 只由 5 层网络，所以它是学习如何构建卷积网络的最佳起点。LeNet5 的架构如下

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-11-28/51381705.jpg)
</center>

LeNet5包含5层网络：

- 第一层：卷积层，该卷积层使用Sigmoid激活函数，并且在后面带有平均池化层；
- 第二层：卷积层，该卷积层使用Sigmoid激活函数，并且在后面带有平均池化层；
- 第三层：全连接层，该全连接层使用Sigmoid激活函数；
- 第四层：全连接层，该全连接层使用Sigmoid激活函数；
- 第五层：输出层。

这意味着我们需要构建 5 个权重和偏置项矩阵，我们模型的主体大概需要 12 行代码完成（5 个神经网络层级、2 个池化层、4 个激活函数还有 1 个 flatten 层）。因为代码比较多，所以我们最好在计算图之外就定义好独立的函数：


```python
# 导入相关包
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```


```python
import numpy as np
```


```python
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffed_dataset = dataset[permutation, :, :]
    shuffed_labels = labels[permutation]
    return shuffed_dataset, shuffed_labels

def one_hot_encode(np_array):
    return (np.arange(10) == np_array[:, None]).astype(np.float32)

def reformat_data(dataset, labels, image_width, image_height, image_depth):
    np_dataset_ = np.array([np.array(image_data).reshape(image_width, image_height, image_depth) for image_data in dataset])
    np_labels_ = one_hot_encode(np.array(labels,dtype=np.float32))
    np_dataset, np_labels = randomize(np_dataset_, np_labels_)
    return np_dataset, np_labels

def flatten_tf_array(array):
    shape = array.get_shape.as_list()
    return tf.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
```

- 导入数据集


```python
%pwd
```




    'E:\\jupyter notebook'




```python
################################### 导入 mnist 数据集 ################################
from mnist import MNIST
```


```python
mnist_folder = '.\data\mnist'
mnist_image_width = 28
mnist_image_height = 28
mnist_image_depth = 1
mnist_num_labels = 10
mndata = MNIST(mnist_folder)
```


```python
# 导入训练集和测试集
mnist_train_dataset_, mnist_train_labels_ = mndata.load_training()
mnist_test_dataset_, mnist_test_labels_ = mndata.load_testing()
```


```python
# 重新改变数据集的形状
mnist_train_dataset, mnist_train_labels = reformat_data(mnist_train_dataset_, mnist_train_labels_, mnist_image_width, mnist_image_height, mnist_image_depth)
mnist_test_dataset, mnist_test_labels = reformat_data(mnist_test_dataset_, mnist_test_labels_, mnist_image_width, mnist_image_height, mnist_image_depth)
```


```python
# 打印所导入数据集的相关信息
print("There are {} images, each of size {}.".format(len(mnist_train_dataset), len(mnist_train_dataset[0])))
print("Meaning each image has the size of 28*28*1 = {}.".format(mnist_image_width * mnist_image_height * 1))
print("The training set contains the following {} labels: {}.".format(len(np.unique(mnist_train_labels_)), np.unique(mnist_train_labels_)))

print("Training set shape ", mnist_train_dataset.shape, mnist_train_labels.shape)
print("Test set shape ", mnist_test_dataset.shape, mnist_test_labels.shape)
```

    There are 60000 images, each of size 28.
    Meaning each image has the size of 28*28*1 = 784.
    The training set contains the following 10 labels: [0 1 2 3 4 5 6 7 8 9].
    Training set shape  (60000, 28, 28, 1) (60000, 10)
    Test set shape  (10000, 28, 28, 1) (10000, 10)
    


```python
from PIL import Image
import matplotlib.pyplot as plt
```


```python
# 显示图像
im = mnist_train_dataset[0].reshape([28,28])
image = Image.fromarray(im)
plt.imshow(image)
plt.show()
```


![](http://otue1rxl3.bkt.clouddn.com/17-11-28/39600002.jpg)



```python
train_dataset_mnist, train_labels_mnist = mnist_train_dataset, mnist_train_labels
test_dataset_mnist, test_labels_mnist = mnist_test_dataset, mnist_test_labels
```


```python
LENET5_BATCH_SIZE = 32
LENET5_PATCH_SIZE = 5
LENET5_PATCH_DEPTH_1 = 6
LENET5_PATCH_DEPTH_2 = 16
LENET5_NUM_HIDDEN_1 = 120
LENET5_NUM_HIDDEN_2 = 84
```


```python
def variabels_lenet5(patch_size = LENET5_PATCH_SIZE,
                     patch_depth1 = LENET5_PATCH_DEPTH_1,
                     patch_depth2 = LENET5_PATCH_DEPTH_2,
                     num_hidden1 = LENET5_NUM_HIDDEN_1,
                     num_hidden2 = LENET5_NUM_HIDDEN_2,
                     image_depth = 1, num_labels = 10):
    w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, image_depth, patch_depth1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([patch_depth1]))
 
    w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, patch_depth1, patch_depth2], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[patch_depth2]))
    
    w3 = tf.Variable(tf.truncated_normal([5*5*patch_depth2, num_hidden1], stddev=0.1))
    b3 = tf.Variable(tf.constant(1.0, shape=[num_hidden1]))
    
    w4 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape=[num_hidden2]))
    
    w5 = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=0.1))
    b5 = tf.Variable(tf.constant(0.1, shape=[num_labels]))
    
    variables = {
        'w1':w1, 'w2':w2, 'w3':w3, 'w4':w4, 'w5':w5,
        'b1':b1, 'b2':b2, 'b3':b3, 'b4':b4, 'b5':b5
    }
    return variables
```


```python
def model_lenet5(data, variables):
    layer1_conv = tf.nn.conv2d(data, variables['w1'], [1,1,1,1], padding='SAME')
    layer1_actv = tf.sigmoid(layer1_conv + variables['b1'])
    layer1_pool = tf.nn.avg_pool(layer1_actv, [1,2,2,1], [1,2,2,1], padding='SAME')
    
    layer2_conv = tf.nn.conv2d(layer1_pool, variables['w2'], [1,1,1,1], padding='VALID')
    layer2_actv = tf.sigmoid(layer2_conv + variables['b2'])
    layer2_pool = tf.nn.avg_pool(layer2_actv, [1,2,2,1], [1,2,2,1], padding='SAME')
    
    # print(layer2_pool.shape.as_list())
    
    shape = layer2_pool.shape.as_list()
    layer2_pool = tf.reshape(layer2_pool, [shape[0], shape[1] * shape[2] * shape[3]])
    flat_layer = layer2_pool
    layer3_fccd = tf.matmul(flat_layer, variables['w3']) + variables['b3']
    layer3_actv = tf.nn.sigmoid(layer3_fccd)
    
    layer4_fccd = tf.matmul(layer3_actv, variables['w4']) + variables['b4']
    layer4_actv = tf.nn.sigmoid(layer4_fccd)
    logits = tf.matmul(layer4_actv, variables['w5']) + variables['b5']
    return logits
```

通过上面独立定义的变量和模型，我们可以一点点调整数据流图而不像前面的全连接网络那样


```python
# 确定模型尺寸的参数
image_width = mnist_image_width
image_height = mnist_image_height
image_depth = mnist_image_depth
num_labels = mnist_num_labels

# 数据集
train_dataset = mnist_train_dataset
train_labels = mnist_train_labels
test_dataset = mnist_test_dataset
test_labels = mnist_test_labels

# 迭代次数和学习速率
num_steps = 10001
display_step = 1000
learning_rate = 0.1
batch_size = 64

graph = tf.Graph()
with graph.as_default():
    # 将数据以TensorFlow友好的形式输入
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, image_depth))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset, tf.float32)
    
    # 初始化权重矩阵和偏差向量
    variabels = variabels_lenet5(image_depth=image_depth, num_labels=num_labels)
    
    # 模型用来计算logit
    model = model_lenet5
    logits = model(tf_train_dataset, variabels)
    
    # 计算softmax交叉熵
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    
    # 梯度下降优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    # 预测
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset, variabels))
```


```python
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized with learning_rate', learning_rate)
    for step in range(num_steps):
        #Since we are using stochastic gradient descent, we are selecting  small batches from the training dataset,
        #and training the convolutional neural network each time with a batch.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset: (offset + batch_size), :, :, :]
        batch_labels = train_labels[offset: (offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        
        if step % display_step == 0:
            train_accuracy = accuracy(predictions, batch_labels)
            test_accuracy = accuracy(test_prediction.eval(), test_labels)
            message = "step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on test set {:02.2f} %".format(step, l, train_accuracy, test_accuracy)
            print(message)
```

    Initialized with learning_rate 0.1
    step 0000 : loss is 002.51, accuracy on training set 15.62 %, accuracy on test set 11.35 %
    step 1000 : loss is 002.25, accuracy on training set 32.81 %, accuracy on test set 23.03 %
    step 2000 : loss is 000.45, accuracy on training set 89.06 %, accuracy on test set 82.01 %
    step 3000 : loss is 000.28, accuracy on training set 96.88 %, accuracy on test set 89.61 %
    step 4000 : loss is 000.25, accuracy on training set 95.31 %, accuracy on test set 92.26 %
    step 5000 : loss is 000.13, accuracy on training set 96.88 %, accuracy on test set 93.52 %
    step 6000 : loss is 000.26, accuracy on training set 90.62 %, accuracy on test set 94.57 %
    step 7000 : loss is 000.20, accuracy on training set 95.31 %, accuracy on test set 95.42 %
    step 8000 : loss is 000.09, accuracy on training set 98.44 %, accuracy on test set 95.87 %
    step 9000 : loss is 000.07, accuracy on training set 100.00 %, accuracy on test set 96.25 %
    step 10000 : loss is 000.12, accuracy on training set 95.31 %, accuracy on test set 96.56 %
    

上面是我重现原作者代码所得到的结果，可见程序是没问题的，可以运行，下面是原作者给出的结果,我把batch_size改成64了，可见效果还提升了点。

```python
>>> Initialized with learning_rate 0.1
>>> step 0000 : loss is 002.49, accuracy on training set 3.12 %, accuracy on test set 10.09 %
>>> step 1000 : loss is 002.29, accuracy on training set 21.88 %, accuracy on test set 9.58 %
>>> step 2000 : loss is 000.73, accuracy on training set 75.00 %, accuracy on test set 78.20 %
>>> step 3000 : loss is 000.41, accuracy on training set 81.25 %, accuracy on test set 86.87 %
>>> step 4000 : loss is 000.26, accuracy on training set 93.75 %, accuracy on test set 90.49 %
>>> step 5000 : loss is 000.28, accuracy on training set 87.50 %, accuracy on test set 92.79 %
>>> step 6000 : loss is 000.23, accuracy on training set 96.88 %, accuracy on test set 93.64 %
>>> step 7000 : loss is 000.18, accuracy on training set 90.62 %, accuracy on test set 95.14 %
>>> step 8000 : loss is 000.14, accuracy on training set 96.88 %, accuracy on test set 95.80 %
>>> step 9000 : loss is 000.35, accuracy on training set 90.62 %, accuracy on test set 96.33 %
>>> step 10000 : loss is 000.12, accuracy on training set 93.75 %, accuracy on test set 96.76 %
```

## 2.6 超参数如何影响一层网络的输出尺寸

一般来说，确实是层级越多神经网络的性能就越好。我们可以添加更多的层级、更改激活函数和池化层、改变学习率并查看每一步对性能的影响。因为层级 i 的输出是层级 i+1 的输入，所以我们需要知道第 i 层神经网络的超参数如何影响其输出尺寸。

为了理解这一点我们需要讨论一下 conv2d() 函数。
>```python
conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=True,
    data_format='NHWC',
    name=None
)
```

>- input: A Tensor. Must be one of the following types: half, float32. A 4-D tensor. The dimension order is interpreted according to the value of data_format, see below for details.
- filter: A Tensor. Must have the same type as input. A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
- strides: A list of ints. 1-D tensor of length 4. The stride of the sliding window for each dimension of input. The dimension order is determined by the value of data_format, see below for details.
- padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
- use_cudnn_on_gpu: An optional bool. Defaults to True.
- data_format: An optional string from: "NHWC", "NCHW". Defaults to "NHWC". Specify the data format of the input and output data. With the -  - default format "NHWC", the data is stored in the order of: [batch, height, width, channels]. Alternatively, the format could be "NCHW", the - data storage order of: [batch, channels, height, width].
- name: A name for the operation (optional).

该函数有四个参数：

- 输入图像，即一个四维张量 [batch size, image_width, image_height, image_depth]
- 权重矩阵，即一个四维张量 [filter_size, filter_size, image_depth, filter_depth]
- 每一个维度的步幅数
- Padding (= 'SAME' / 'VALID')

这四个参数决定了输出图像的尺寸。前面两个参数都是四维张量，其包括了批量输入图像的信息和卷积滤波器的权值。

第三个参数为卷积的步幅（stride），即卷积滤波器在 4 个维度中的每一次移动的距离。四个中间的第一个维度代表着图像的批量数，这个维度肯定每次只能移动一张图片。最后一个维度为图片深度（即色彩通道数，1 代表灰度图片，而 3 代表 RGB 图片），因为我们通常并不想跳过任何一个通道，所以这一个值也通常为 1。第二个和第三个维度代表 X 和 Y 方向（图片宽度和高度）的步幅。如果我们希望能应用步幅参数，我们需要设定每个维度的移动步幅。例如设定步幅为 1，那么步幅参数就需要设定为 [1, 1, 1, 1]，如果我们希望在图像上移动的步幅设定为 2，步幅参数为 [1, 2, 2, 1]。

最后一个参数表明 TensorFlow 是否需要使用 0 来填补图像周边，这样以确保图像输出尺寸在步幅参数设定为 1 的情况下保持不变。通过设置 padding = 'SAME'，图像会只使用 0 来填补周边（输出尺寸不变），而 padding = 'VALID'则不会使用 0。在下图中，我们将看到两个使用卷积滤波器在图像上扫描的案例，其中滤波器的大小为 5 x 5、图像的大小为 28 x 28。左边的 Padding 参数设置为'SAME'，并且最后四行/列的信息也会包含在输出图像中。而右边 padding 设置为 'VALID'，最后四行/列是不包括在输出图像内的。
<center>![](http://otue1rxl3.bkt.clouddn.com/17-11-28/88148369.jpg)</center>
没有 padding 的图片，最后四个像素点是无法包含在内的，因为卷积滤波器已经移动到了图片的边缘。这就意味着输入 28 x 28 尺寸的图片，输出尺寸只有 24 x 24。如果 padding = 'SAME'，那么输出尺寸就是 28 x 28。
如果我们输入图片尺寸是 28 x 28、滤波器尺寸为 5 x 5，步幅分别设置为 1 到 4，那么就能得到下表
<center>![](http://otue1rxl3.bkt.clouddn.com/17-11-28/13366246.jpg)</center>
由此可见，对于stride为1，用0补充和不用0补充的输出图像大小分别为28x28和24x24,如果stride为2，则为14x14和12x12，如果stride为3，则为10x10和8x8等。

对于任意给定的步幅 S、滤波器尺寸 K、图像尺寸 W、padding 尺寸 P，输出的图像尺寸可以总结上表的规则如下：

<center>O = 1+(W-K+2P)/S</center>

## 2.7 调整 LeNet5 架构
LeNet5 架构在原论文中使用的是 Sigmoid 激活函数和平均池化。然而如今神经网络使用 ReLU 激活函数更为常见。所以我们可以修改一下 LeNet5 架构，并看看是否能获得性能上的提升，我们可以称这种修改的架构为类 LeNet5 架构。


```python
LENET5_LIKE_BATCH_SIZE = 32
LENET5_LIKE_FILTER_SIZE = 5
LENET5_LIKE_FILTER_DEPTH = 16
LENET5_LIKE_NUM_HIDDEN = 120

def variables_lenet5_like(filter_size = LENET5_LIKE_FILTER_SIZE, 
                          filter_depth = LENET5_LIKE_FILTER_DEPTH, 
                          num_hidden = LENET5_LIKE_NUM_HIDDEN,
                          image_width = 28, image_depth = 1, num_labels = 10):
 
    w1 = tf.Variable(tf.truncated_normal([filter_size, filter_size, image_depth, filter_depth], stddev=0.1))
    b1 = tf.Variable(tf.zeros([filter_depth]))

    w2 = tf.Variable(tf.truncated_normal([filter_size, filter_size, filter_depth, filter_depth], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[filter_depth]))
 
    w3 = tf.Variable(tf.truncated_normal([(image_width // 4)*(image_width // 4)*filter_depth , num_hidden], stddev=0.1))
    b3 = tf.Variable(tf.constant(1.0, shape = [num_hidden]))

    w4 = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape = [num_hidden]))
 
    w5 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    b5 = tf.Variable(tf.constant(1.0, shape = [num_labels]))
    variables = {
                  'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5,
                  'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5
                }
    return variables

def model_lenet5_like(data, variables):
    layer1_conv = tf.nn.conv2d(data, variables['w1'], [1, 1, 1, 1], padding='SAME')
    layer1_actv = tf.nn.relu(layer1_conv + variables['b1'])
    layer1_pool = tf.nn.avg_pool(layer1_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    layer2_conv = tf.nn.conv2d(layer1_pool, variables['w2'], [1, 1, 1, 1], padding='SAME')
    layer2_actv = tf.nn.relu(layer2_conv + variables['b2'])
    layer2_pool = tf.nn.avg_pool(layer2_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    shape = layer2_pool.shape.as_list()
    layer2_pool = tf.reshape(layer2_pool, [shape[0], shape[1] * shape[2] * shape[3]])
    flat_layer = layer2_pool
    # flat_layer = flatten_tf_array(layer2_pool)
    layer3_fccd = tf.matmul(flat_layer, variables['w3']) + variables['b3']
    layer3_actv = tf.nn.relu(layer3_fccd)
    #layer3_drop = tf.nn.dropout(layer3_actv, 0.5)
 
    layer4_fccd = tf.matmul(layer3_actv, variables['w4']) + variables['b4']
    layer4_actv = tf.nn.relu(layer4_fccd)
   #layer4_drop = tf.nn.dropout(layer4_actv, 0.5)
 
    logits = tf.matmul(layer4_actv, variables['w5']) + variables['b5']
    return logits
```

主要区别在于我们使用relu激活函数而不是sigmoid激活。 

除了激活函数之外，我们还可以更改已使用的优化器，以查看不同优化器对精度的影响。