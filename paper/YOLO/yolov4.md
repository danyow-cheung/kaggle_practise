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



### Bag of specials



