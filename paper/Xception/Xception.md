[toc]



# Xception论文记录和代码复现



## Abstract

<u>Inception 模块</u>和<u>深度可分离卷积操作</u>在卷积神经网络中作为常规卷积之间的中间步骤。

從這個角度來看，深度可分離卷積可以理解為具有最大塔數的Inception modules。



X ception 和Inception V3有相同的参数



## Introduction

深度学习已经是计算机视觉中主要的算法解决方法。

卷积神经网络的历史是从[LeNe](https://zh.d2l.ai/chapter_convolutional-neural-networks/lenet.html)t样式的模型，<u>*which were simple stacks of convolutions for feature extraction and max-pooling operations for spatial sub-sampling*.</u>（它们是用于特征提取的**简单卷积堆栈和用于空间子采样的最大池操作**。）在这种模型中，卷积操作重复多次在maxpooling之间，这样可以允许网络在每个空间尺度上学习更丰富的特征。这也使得网络层数越来越深。

接着在2014年出现了Inception结构。



基本Inception模块的构造有不同的版本。

虽然Inception 模块类似于卷积特征提取器，但是Inception凭借经验用更少的参数学习特征

### The Inception hypothesis 

一层卷积层尝试在3D空间中，使用2个空间维度（宽度和高度）和一个通道维度，因此一个卷积层同时映射**<u>跨通道相关性</u>**和**<u>空间相关性</u>**。



this idea behind the inception module is to make process easier and more efficient by explicitly factoring.(显式因式分解)



in effect,the fundamental hypothesis behind Inception is that cross-channel correlations and spatial correlations are sufficiently decoupled (解耦) that it is preferable not to map them jointly.



### The continuum between convolutions and separable convolutioins

a depthwise separable convolution ,commonly called 'separable convolution' in deep learning framework.

consists in a depthwise convolution, 

**a spatial convolution performed independently over each channel of an input,followed by a <u>pointwise convolution逐点卷积</u>**

在輸入的每個通道上獨立執行空間卷積，然後是<u>逐點卷積</u>



# 代码复现

> - https://towardsdatascience.com/xception-from-scratch-using-tensorflow-even-better-than-inception-940fb231ced9
>
> - https://stephan-osterburg.gitbook.io/coding/coding/ml-dl/tensorfow/ch3-xception/implementation-of-xception-model

# 引用

- [Xception:Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf)

- https://towardsdatascience.com/xception-from-scratch-using-tensorflow-even-better-than-inception-940fb231ced9

- https://stephan-osterburg.gitbook.io/coding/coding/ml-dl/tensorfow/ch3-xception/implementation-of-xception-model
