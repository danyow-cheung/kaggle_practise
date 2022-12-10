# Xception: Deep Learning with Depthwise Separable Convolutions具有深度可分卷积的深度学习



> 读论文的笔记
>
> https://arxiv.org/pdf/1610.02357.pdf



# 1.Introduction

深度学习已经是计算机视觉中主要的算法解决方法。

卷积神经网络的历史是从[LeNe](https://zh.d2l.ai/chapter_convolutional-neural-networks/lenet.html)t样式的模型，<u>*which were simple stacks of convolutions for feature extraction and max-pooling operations for spatial sub-sampling*.</u>（它们是用于特征提取的简单卷积堆栈和用于空间子采样的最大池操作。）在这种模型中，卷积操作重复多次在maxpooling之间，这样可以允许网络在每个空间尺度上学习更丰富的特征。这也使得网络层数越来越深。



