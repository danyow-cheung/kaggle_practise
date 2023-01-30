# YOLOv4:Optimal Speed and Accuracy of Object Detection

## Abstract 

we assume that such universal features include Weighted-Residual-Connections(WRC).Cross-Stage-Partial-connections (CSP),





## Introduction

the majority of CNN-based object detectors are largely applicable only for recommendation systems

基於CNN的目標檢測很大程度上取決了操作系統。





Improving the real-time object detector accuracy enables using them not only for hint generating recommendation systems,but also for stand-alone process management and human input reduction

通過改進實時對象檢測器，不僅可以將其用於生成推薦系統，還可以用於獨立的過程管理和減少人工輸入。

The most accurate modern neural networks do not operate in real time and require large number of GPUs for training with a large mini-batch-size 

最精確的現代神經網路不能實时運行，並且需要大量GPU來進行大規模的小批量訓練



The main goal of this work is <u>**designing a fast operating speed of an object detector in production systems and optimization for parallel compuations**</u>

設計生產系統中物體監測器的快速運行速度並優化並行計算，而不是低體積的理論指標。



## Related work

### Object detection models

a modern detector is usually composed of two parts,a backbone which is pre-trained on ImageNet and a head which is used to predict classes and bounding boxes of objects 

現代目標檢測分為倆個步驟1⃣️backbone預訓練模型2⃣️head預測類別概率



關於head的部分有

- one-stage object detector單通道檢測器

  > yolo,ssd

- two-stage object detector雙通道檢測器 

  > R-CNN series



It is also possible to make a two-stage object detector an anchor-free object detector 

雙通道也能發展成anchor-free





Object detectors developed in recent years often insert some layers between backbone and head,and these layers are usually used to collect feature maps from different stages.

神經網絡通過增加網絡層數來手機不同角度的特徵。

we can call it the neck of the an object detector

我們將這個稱為網絡的頸部

Usually,a neck is composed of serveral bottom-up paths and serveral top-down paths

目標檢測的頸部通常有自下而上，或自上而下的路徑組成。



To sum up,an ordinary object detector is composed of serval parts

**<u>總計來說，目標檢測由以下組成</u>**

<img src= 'object-detector-model.png'>

- **Input:**Image,Patches,Image Pyramid

- **Backbone:**(VGG16,ResNet-50,SpineNet)(gpu platform),EfficientNet-B0/B7,CSPResNeXt50,CSPDarkNet53

- **Neck:**

  *collect features map from different stages*

  - **Additional blocks:**SPP,ASPP,RFB,SAM
  - **Path-aggregation blocks:**PPN,PAN,NAS-FPN,Fully-connected FPN,BiFPN,ASFF,SFAM

- **Heads:**

  - **Dense Prediction(one-stage):**
    - RPN,SSD,YOLO,RetinaNet(anchor based)
    - CornetNet,CenterNet,MatrixNet,FCOS,(anchor free)
  - **Sparse Prediction(two-stage):**
    - Faster R-CNN,R-FCN,Mask R-CNN(anchor based)
    - RepPoints(anchor free)





### Bag of freebies

> usually  a convolutional object detector is trained offline.Therefore,researchers always like to take this advantage and <u>develop receive better accuracy without increasing the inference cost</u>使用更好的訓練方法而不增加推理成本

<u>**we can call these methods that only change the training strategy or only increase the training cost as 'bag of freebies'**</u>

這種只改變訓練策略，或者增加訓練成本的方法，稱呼為bag of freebies免費包



what is often adopted by object detection methods and meets the definition of bag of freebies is data augmentation.

对象检测方法经常采用的并且符合freebies包定义的是数据增强。



The purpose of data augmentation is to increase the variablity of the input images ,so that the designed object detection model has higher robustness to the images obtained from different enviroments

數據增強的目標是在不同環境下，模型有更高的魯棒性。

photometric distoritions ,geometric distoritions 

常見的數據增強方法有photometric distoritions光度失真 ,geometric distoritions幾何失真



the data augmentation methods mentioned above are all <u>pixel-wise adjustments</u> .逐像素調整



>  in addition,some researchers engaged in data augmentation put their emphasis on simulating object occlusion issues.將重點放在模擬對象遮擋的發布者上

>  in addition to above mentioned methods,style transfer GAN is also used for data augmentation,and such usuages can effectively reduce the texture bias learned by CNN甚至用GAN 產生的圖片來訓練CNN





Different from the various approaches proposed above,some other bag of freebies methods are dedicated to solving the problem that the semantic distribution in the datase may have bias 

致力解決數據集中語義分佈存在偏差的問題，

 

In dealing with the problem of semantic distribution bias,a very important issue is that there is a problem of data imbalance between different classessm,and this problem is often solved by hard negative example mining or online hard example mining in two-stage object detector 

不同類之間存在數據不平衡的問題，這一問題通常通過雙通道檢測中的硬否定示例挖掘或在線硬示例挖掘解決。



。。。。



### Bag of specials

for those plugin modules and post-processing methods that **<u>only increase the inference cost by a small amount but can signifficantly improve the accuracy of object detection.</u>**we can call them 'bag of specials'

增加小部分的推理時間，但明顯增加模型精度的預處理方法



Generally speaking these plugin modules are for enhancing certain attributes in these plugin modules are for enhancing certain attributes in a model ,such as <u>enlarging receptive field,introducing attention mechasim,or strengthening feature intergration catpability</u> 

常見的方法有增大感受嘢，注意力機制等



