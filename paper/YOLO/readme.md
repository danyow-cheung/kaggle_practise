# You Only Look Once,Unified ,Real-Time Object Detection

## Abstract

we frame object detection <u>**as a regression problem to spatially separated bouding boxes and associated probabilities**</u>

作為空間分離的邊界框和相關概率的回歸問題



## Introduction

Current detection systems repurpose classifiers to perform detection.To detect an object, these 

systems take a classifier for that object and **evaluate it at various locations and scales in a test image.** System like <u>deformable parts models</u>(DPM) **use a sliding window approach where the classifier is run at evenly spaced locations over the entire image**

當前的檢測系統重新利用分類器來執行檢測。為了檢測物體，這些系統採用該對象的分類器，並在測試**圖像中的不同位置和尺度上對其進行評估**。 類似<u>可變形部件模型</u>(DPM) 的系統**使用滑動窗口方法，其中分類器在整個圖像上均勻分佈的位置運行。**



其他像R-CNN的（**<u>region proposal</u>**）<u>區域提議方法</u>的實現步驟是

1. first generate potential bounding boxes in an image and then run a classififer on these proposed boxes首先在圖像中生成潛在的邊界框，然後在這些提議的框上運行分類器
2. classification 識別檢測
3. post-processing is used to refine the bouding boxes,eliminate duplicate detections and rescore the boxes based on other objects in the scene.後處理用於細化邊界框，消除重複檢測並根據場景中的其他對像對邊界框重新評分。

總結,these complex pipelines are slow and hard to optimize because each individual component must be trained separately這些複雜的管道很慢且難以優化，因為每個單獨的組件都必須單獨訓練



> hier ist yolo

**we reframe object detection as a single regression problem,straight from pixel to bouding box  coordinates and class probabilities**

我們將目標檢測重新定義為一個單一的回歸問題，直接從像素到邊界框坐標和類別概率