#  Faster R-CNN:Towards Real-Time Object Detection with Region Proposal Networks

> RPN:region proposal  network 区域建议网络 

## Introduction

Although region-based CNNs were computationally expensive as originally developed in [5], their cost has been drastically reduced thanks to sharing convolutions 



尽管基于区域的神经网络在计算上比最初在[5]中开发的要昂贵，但由<u>于共享卷积</u>，它们的成本已经大大降低



Fast R-CNN [2], achieves near real-time rates using very deep networks [3], when ignoring the time spent on region proposals. Now, proposals are the test-time computational bottleneck in state-of-the-art detection systems

Fast R-CNN[2]，使用非常深的网络[3]实现了接近实时的速率，而忽略了在区域提案上花费的时间。现在，<u>proposals是最先进检测系统的测试时间计算瓶颈</u>



Region proposal methods typically rely on inexpensive features and economical inference schemes

区域建议方法通常依赖于廉价的特征和经济的推理方案







Selective Search [4], one of the most popular methods, greedily merges superpixels based on engineered low-level features

选择性搜索[4]是最流行的方法之一，它基于设计的低级特征贪婪地合并超级像素



An obvious way to accelerate proposal computation is to reimplement it for the GPU 

基于CNN的深度学习建议推广到GPU来加快proposal的计算



. This may be an effective engineering solution, but re-implementation ignores the down-stream detection network and therefore misses important opportunities for sharing computation.

这可能是一种有效的工程解决方案，但重新实现忽略了下游检测网络，因此错过了共享计算的重要机会。







In this paper, we show that an algorithmic change— computing proposals with a deep convolutional neural network—leads to an elegant and effective solution where proposal computation is nearly cost-free given the detection network’s computation

在本文中，我们证明了一种使用深度卷积神经网络的算法变化计算建议可以产生一种优雅而有效的解决方案，其中，在给定检测网络的计算的情况下，proposal计算几乎是无成本的





Our observation is that the convolutional feature maps used by region-based detectors, like Fast RCNN, can also be used for generating region proposals

我们的观察结果是，基于区域的检测器（如Fast RCNN）使用的卷积特征图也可以用于生成区域建议



On top of these convolutional features, we construct an RPN by adding a few additional convolutional layers that simultaneously regress region bounds and objectness scores at each location on a regular grid.

方法：在这些卷积特征的基础上，我们通过添加几个额外的卷积层来构建RPN，这些卷积层同时做回归<u>规则网格上每个位置的区域边界</u>和<u>对象性得分</u>。



RPNs are designed to efficiently predict region proposals with a wide range of scales and aspect ratios. In contrast to prevalent methods，that use  pyramids of images (Figure 1, a) or pyramids of filters (Figure 1, b), we introduce novel “anchor” boxes that serve as references at multiple scales and aspect ratios. 

RPN旨在有效预测具有广泛规模和纵横比的区域方案。与使用图像金字塔（图1，a）或过滤器金字塔（图2，b）的流行方法不同，我们引入了新的“锚”框，以多种比例和纵横比作为参考。

<img src = 'figure2.png'>

<u>**Our scheme can be thought of as a pyramid of regression references (Figure 1, c), which avoids enumerating images or filters of multiple scales or aspect ratios. This model performs well when trained and tested using single-scale images and thus benefits running speed**</u> 

我们的方案可以被认为是回归参考的金字塔（图1，c），它避免了枚举多个尺度或纵横比的图像或过滤器。当使用单尺度图像进行训练和测试时，该模型表现良好，从而提高了运行速度

Meanwhile, our method waives nearly all computational burdens of Selective Search at test-time—the effective running time for proposals is just 10 milliseconds. 

 同时，我们的方法在测试时几乎免除了选择性搜索的所有计算负担，提案的有效运行时间仅为10毫秒。



## RELATED WORK

### Object Proposals 

目标建议

Widely used object proposal methods include those based on grouping super-pixels (e.g., Selective Search [4], CPMC [22], MCG [23]) and those based on sliding windows (e.g., objectness in windows [24], EdgeBoxes [6])

广泛使用的对象建议方法包括基于分组超像素的方法（例如，选择性搜索[4]、CPMC[22]、MCG[23]）和基于滑动窗口的方法（如，窗口中的对象性[24]、边缘盒[6]）









### Deep Networks for Object Detection

R-CNN mainly plays as a classifier, and it does not predict object bounds (except for refining by bounding box regression). Its accuracy depends on the performance of <u>the region proposal module</u> 

R-CNN主要用作分类器，它不预测对象边界（通过边界框回归进行细化除外）。其准确性取决于<u>区域提案模块的性能</u>



In the **OverFeat** method [9], a fully-connected layer is trained to predict the box coordinates for the localization task that assumes a single object. The fully-connected layer is then turned into a convolutional layer for detecting multiple class-specific objects .

在OverFeat方法[9]中，训练<u>完全连接（Fully connected ）</u>的层以预测假设单个对象的定位任务的框坐标。然后将完全连接的层转换为卷积层，用于检测多个类特定对象。

 

The **MultiBox** methods [26], [27] generate region proposals from a network whose last fully-connected layer simultaneously predicts multiple class-agnostic boxes, generalizing the “singlebox” fashion of OverFeat.

MultiBox方法[26]，[27]从网络生成区域建议，该网络的最后一个完全连接的层同时预测多个类不可知的框，概括了OverFeat的“单框”方式。







Shared computation of convolutions [9], [1], [29], [7], [2] has been attracting increasing attention for efficient, yet accurate, visual recognition.

卷积的共享计算[9]，[1]，[29]，[7]，[2]对于高效但准确的视觉识别已经引起了越来越多的关注。





## FASTER-RCNN

Our object detection system, called Faster R-CNN, is composed of two modules. The first module is a deep fully convolutional network that proposes regions, and the second module is the Fast R-CNN detector [2] that uses the proposed regions.

我们的物体检测系统叫做Faster R-CNN由两个模块组成。第一个模块是
提出区域的全卷积网络，第二模块是快速R-CNN检测器[2].使用建议的区域。

<img src='figure2-1.png'>

faster r-cnn是单通道统一的网络，rpn作为同一网络的attention

the RPN module tells the Fast R-CNN module where to look.

RPN模块告诉网络应该去看哪一部分的区域





### Region Proposal Networks

A Region Proposal Network (RPN) takes an image (of any size) as input and outputs a set of rectangular object proposals, each with an objectness score

区域建议网络（RPN）将图像（任何大小）作为输入，并输出一组矩形对象建议，每个建议都有一个对象得分





Because our ultimate goal is to share computation with a Fast R-CNN object detection network [2], we assume that both nets share a common set of convolutional layers

因为我们的最终目标是与快速R-CNN对象检测网络共享计算[2]，我们假设两个网络共享一组共同的卷积层





To generate region proposals, we slide a small network over the convolutional feature map output by the last shared convolutional layer. 

为了生成区域建议，我们在最后一个共享卷积层输出的卷积特征图上滑动一个小网络。

 <u>**This small network takes as input an `n × n` spatial window of the input convolutional feature map.**</u> 

这个小网络将输入卷积特征图的“n×n”空间窗口作为输入。



Each sliding window is mapped to a lower-dimensional feature (256-d for ZF and 512-d for VGG, with ReLU [33] following).





This feature is fed into two sibling fullyconnected layers—a box-regression layer (reg) and a box-classification layer (cls). 

这个特性被输入到两个兄弟完全连接的层中——一个box回归层（reg）和一个box分类层（cls）。



#### Anchors

At each sliding-window location, we simultaneously predict multiple region proposals, where the number of maximum possible proposals for each location is denoted as k

在每个滑动窗口位置，我们同时预测多个区域建议，其中每个位置的最大可能建议数表示为k

. An anchor is centered at the sliding window in question, and is associated with a scale and aspect ratio 

。锚定位于所讨论的滑动窗口的中心.并且与比例和纵横比相关联





#### Translation-Invariant Anchors

平移不变anchors 



As a comparison, the MultiBox method [27] uses k-means to generate 800 anchors, which are not translation invariant. So <u>MultiBox does not guarantee that the same proposal is generated if an object is translated.</u>

作为比较，MultiBox方法[27]使用k均值来生成800个锚，
它们不是平移不变的。因此，MultiBox不能保证在翻译对象时生成相同的建议。

The translation-invariant property also reduces the model size. 

减小模型的大小







#### Multi-Scale Anchors as Regressioin References 

多尺度锚框作为回归参考



there have been two popular ways for multi-scale predictions. 

- The first way is based on image/feature pyramids, e.g., in DPM [8] and CNNbased methods [9], [1], [2]

  基于图像或特征金字塔



- The second way is to use sliding windows of multiple scales (and/or aspect ratios) on the feature maps. 

  第二种方法是在特征地图上使用多个比例（和/或纵横比）的滑动窗口。



As a comparison, our anchor-based method is built on a pyramid of anchors, which is more cost-efficient.

作为比较，我们的基于锚的方法建立在锚的金字塔上，这更具成本效益。





Because of this multi-scale design based on anchors, we can simply use the convolutional features computed on a single-scale image, as is also done by the Fast R-CNN detector [2]. The design of multiscale anchors is a key component for sharing features without extra cost for addressing scales.

由于这种基于锚的多尺度设计，我们可以简单地使用在单尺度图像上计算的卷积特征，正如Fast R-CNN检测器[2]所做的那样。多尺度锚的设计是共享特征的关键组成部分，而无需额外的尺度寻址成本。





### Loss Function

训练FPNs，指代两种情况的anchor box为positive label

(i) the anchor/anchors with the highest Intersection-overUnion (IoU) overlap with a ground-truth box, or 具有最高联合交叉点（IoU）的锚与地面实况框重叠



(ii) an anchor that has an IoU overlap higher than 0.7 with any ground-truth box. 

与任何地面真值框具有高于0.7的IoU重叠的锚。





. Note that a single ground-truth box may assign positive labels to multiple anchors.

单个ground truth box可能会被指派多个positive label

