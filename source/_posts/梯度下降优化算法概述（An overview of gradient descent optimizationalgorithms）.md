---
layout: post
title: 梯度下降优化算法概述（An overview of gradient descent optimizationalgorithms）
categories: paper
description: author：Sebastian Ruder</br>梯度下降优化算法变得越来越流行，但是它通常被用作黑盒优化，因为关于梯度下降算法的长处和短处很难给出实际的解释。这篇文章的目的是提供给读者关于不同算法的行为一个直观的感受，这些算法在将来读者可能会用到。在这篇概述中，我们着眼于梯度下降算法的不同变体，总结挑战，介绍最常见的优化算法，回顾在并行环境和分布式环境中的结构以及研究梯度下降优化算法一些额外的策略。
date: 2017-10-17 20:11:07
tags: [paper, gradient descent, 综述]

---

# 1 前言
　　梯度下降算法是目前用来进行最优化的最流行的算法之一，并且也是来优化神经网络的最常用的方式。与此同时，每一款先进的深度学习库都包含了优化梯度下降的各种算法的实现（例如：lasagne、caffeine、Keras）。然而，这些算法通常用作黑盒优化器，关于算法的优点和缺点很难给出一个实际的解释。
　　这篇文章的目的就是提供给读者一个关于不同优化梯度下降算法的一个直观感受，这可能会在将来帮助读者使用这些算法。接下来，我们将在第三章简单总结训练过程中的挑战，随后，在第四章中我们将介绍最常见的最优化算法，通过展示它们解决这些挑战的能力和怎样产生更新规则的导数的方式进行介绍。在第五章，我们将简要介绍算法和结构，以优化并行和分布式环境下的梯度下降。最后，在第六章中，我们将介绍能够帮助优化梯度下降算法的一些额外的策略。
　　梯度下降是一种最小化目标函数J(θ)的参数的方式，模型参数θ属于Rd，通过从目标函数▽θJ(θ)的梯度相反的方向更新参数来更新模型的参数。学习率η决定了接近（局部）最小值的步长。也就是说，我们沿着由目标函数所描绘的曲面的斜率方向下降，直到到达一个山谷。
# 2 梯度下降的变体
　　梯度下降有3种不同的变体，它们的不同在于我们使用多大的数据去计算目标函数的梯度。根据数据量的大小，我们权衡了更新参数的准确性和执行参数更新所需的时间。
## 2.1 批量梯度下降（Batch gradient descent）
　　Vanilla梯度下降也叫作批量梯度下降，它计算对于全部的训练集关于参数θ的代价函数的梯度：
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/62906877.jpg)
</center>

　　当我们需要计算整个数据集的梯度来执行一次更新时，批量梯度下降将会变得很慢，并且对于没有装载进内存的数据是非常棘手的。同时批量梯度下降算法不允许在线更新模型，也就是是说实时添加一个新的样本。
　　批量梯度下降的程序代码如下所示：
```python
for i in range(nb_epochs):
	params_grad = evaluate_gradient(loss_function, data, params)
	params = params - learning_rate * params_grad
```
　　对于预定义的epochs的数量，我们首先根据我们的参数向量params计算关于整个数据集损失函数的梯度向量params_grad。注意，最先进的深度学习库提供了自动微分计算，这能根据一些参数有效的计算梯度。如果你自己进行了梯度推导，那么进行梯度检查会是一个不错的主意。
　　接下来我们在梯度方向更新我们的参数，学习率决定了每一步更新的大小。对于凸误差表面批量梯度下降可以保证收敛到全局最小，对于非凸误差表面可以保证收敛到局部最小值。
# 2.2 随机梯度下降（Stochastic gradient descent）
　　相比之下，随机梯度下降（SGD）对每一个训练样本x(i)与标签y(i)都执行参数更新：
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/86369040.jpg)
</center>

　　批量梯度下降对于大数据集执行冗余计算，因为在每一个参数更新之前它重复计算了相同样本的梯度。SGD通过一次执行一次更新去除了这种冗余计算，因此，SGD通常更快，并且可以用于在线学习。SGD执行具有高方差的频繁更新，这造成目标函数具有很大的波动，如图1所示。
　　当批量梯度下降收敛到参数局部的最小值时，一方面，SGD的波动可以使收敛跳出局部收敛到一个新的潜在的更好的局部最小值，另一方面，这最终将收敛至精确的最小值，因为SGD将保持“过火”。然而，已经表明，当我们慢慢降低学习速率时，SGD和批量梯度下降具有相同的收敛性，几乎可以确定的收敛到局部最小或者全局最小，它们分别对应着非凸优化和凸优化的结果。SGD的代码块只是简单的在训练样本上添加了一个循环，并计算关于样本的梯度值。注意，下面这段代码中，每一次循环都对数据集进行了重新洗牌。
```python
for i in range(nb_epochs):
	np.random.shuffle(data)
	for example indata:
		params_grad = evaluate_gradient(loss_function, example, params)
		params = params - learning_rate * params_grad
```

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-19/89582446.jpg)
图1 SGD波动
</center>

# 2.3 小批量梯度下降（Mini-batch gradient descent）
　　小批量梯度下降能获得两全其美的效果，它对每一个有n个训练样本的小批量执行参数更新：
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/11036578.jpg)
</center>

　　用这种方法（a）降低了参数更新的方差，这可以获得一个更加稳定的收敛结果；（b）可以利用高度优化的常见的最先进的深度学习库进行矩阵最优化，这个可以使计算关于小批量的梯度变得非常高效。常见的小批量的大小在50到256之间，但是可以根据不同的应用变化。当训练一个神经网络时，小批量下降算法是经典算法的选择，并且，在小批量使用的同时通常也使用SGD。注意：为了简单起见，修改自文章其他部分的SGD，我们去掉了参数x(i:i+n)和y(i:i+n)。
　　在代码中，为了取代迭代所有的样本，我们将迭代的小批量的大小设置为50：
```python
for i in range(nb_epochs):
	np.random.shuffle(data)
    for batch in get_batches(data, batch_size = 50):
    	params_grad = evaulate_gradient(loss_function, batch, params)
        params = params - learning_rate * params_grad
```
# 3 挑战（Challenges）
　　然而，Vanilla小批量梯度下降不能保证有一个好的收敛结果，但提供了一些需要解决的挑战：
- 选择合适的学习速率是困难的。如果学习速率太小将导致收敛变得非常的缓慢，一旦学习速率太大，则可能会阻碍收敛，并且造成损失函数在最小值附近波动甚至可能导致发散。
- 学习速率表在训练过程试着去降低学习速率，例如：退火，也就是说根据预先设定好的学习速率表减小学习速率，或者是当目标值的变化低于阈值的时候。
- 此外，相同的学习速率适应所有的参数更新。如果我们的数据稀疏，并且数据的特征有非常不同的频率，我们可能不想讲所有的参数更新到相同的程度，但是对很少出现的特征执行大的更新。
- 最小化神经网络常见的非凸误差函数的另一个关键问题是避免被困在其众多次优局部最小值中。Dauphin等人指出困难实际上不是来自局部极小值，而是来自鞍点，也就是说，一个维度向上倾斜，另一个向下倾斜。这些鞍点通常被同一错误的高原所包围，这将使得SGD很难跳出局部最小，因为梯度在这一点的所有纬度上都接近0.

# 4 梯度下降优化算法（Gradient descent optimization algorithms）
　　接下来我们将概述一些算法，这些算法被深度学习团队广泛应用于应对上述挑战。我们不会讨论在实践中对于高维数据集无法计算的算法，例如：二阶方法，如牛顿法。
## 4.1 Momentum
　　SGD在找最优路径的时候存在问题，即在一个维度上的表面曲线比另一个维度陡峭得多，这在局部最优解周围很常见。在这种情况下，SGD在沟壑的斜坡上震荡，仅在接近局部最优解时犹豫不前，如图2(a)所示。
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/30795993.jpg)
图2
</center>

　　Momentum是一种可以帮助在相关方向上加速SGD的方法，并且它可以抑制振荡，如图2(b)所示。它通过增加一个之前更新向量的分数γ迭代至当前的更新向量实现：
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/68238377.jpg)
</center>

　　Momentum的γ通常设置为0.9或者其他相似的值。
　　本质上来说，当使用Momentum时，我们将一个球推向山谷，这个球在滚向山脚的过程中积攒动量，在下降的方向上变得越来越快（如果考虑空气阻力，即γ < 1，直到它达到最终的速度）。这在参数更新的过程中也发生了同样的事：对于梯度指向相同方向的维度，动量在增加，并且对于梯度方向改变的维度动量的更新在削减。因此，我们获得了更快的收敛并抑制了振荡。

## 4.2 Nesterov加速梯度（Nesterov accelerated gradient）
　　然而球沿着山坡滚下山脚的结果是非常不满意的，我们应该有一个智能的球，这个球对于接下来将要去哪有一个概念，因此它可以在山坡再一次向上弯曲时减慢速度。
　　Nesterov加速梯度（NAG）就是给动量参数提供这种先知能力的方法。我们都知道我们将使用动量参数γv(t-1)来改变参数θ，因此，计算θ-γv(t-1)给我们提供了参数的下一个位置的一个近似值，一个粗略的概念关于我们的参数将会是多少。因此，我们可以通过计算梯度更有效的预测以后的结果，计算梯度的过程不是根据我们当前的参数θ，而是根据参数将出现的近似位置：
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/16937378.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/84262650.jpg)
图3 Nesterov 更新
</center>

　　而且，我们将动量参数γ的值设置为0.9左右，当Momentum计算当前的梯度（图3中的蓝色小向量 ），然后在梯度累积更新的方向上大跳一步时（图3蓝色大向量），NAG首先在之前累积的梯度方向datiao

## 4.3 自适应梯度算法（Adagrad）
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/69091829.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/73544408.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/453033.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/85849134.jpg)
</center>

## 4.4 Adadelta

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/93630508.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/34687376.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/97699924.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/34877510.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/2721998.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/55658071.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/55176681.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/57009286.jpg)
</center>

## 4.5 RMSprop

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/57917998.jpg)
</center>

## 4.6 Adam

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/47558734.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/88301127.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/23580140.jpg)
</center>

## 4.7 AdaMax

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/13190077.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/75999735.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/46476309.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/16909009.jpg)
</center>

## 4.8 Nadam

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/15359030.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/45457135.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/91491839.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/81453023.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/9569250.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/76616104.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/14371584.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/67882309.jpg)
</center>

## 4.9 可视化算法（Visualization of algorithms）
<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-18/99650733.jpg)
</center>

## 4.10 使用哪一个优化器？（Which optimizer to use?）

# 5 并行和分布SGD（Parallelizing and distributing SGD）
## 5.1 Hogwild!
## 5.2 Downpour SGD
## 5.3 SGD 时延容忍算法（Delay-tolerant Algorithms for SGD）
## 5.4 TensorFlow
## 5.5 Elastic Averaging SGD
# 6 SGD优化附加策略（Additional strategies for optimizing SGD）
## 6.1 Shuffling and Curriculum Learning
## 6.2 Batch normalization
## 6.3 Early stopping（早停法）
## 6.4 Gradient noise（梯度噪声）

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-19/23032284.jpg)
</center>

<center>
![](http://otue1rxl3.bkt.clouddn.com/17-10-19/76751205.jpg)
</center>

# 7 总结