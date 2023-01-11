# YOLO9000:Better,Faster,Stronger

## Abstract

First,we propose various improvements to the YOLO detection method,both novel and drawn from prior work

YOLO检测方法改进



outperforming state-of-art methods like Faster R-CNN with ResNet and SSD while still running significantly faster.

性能依旧比Faster R-CNN和SSD好



## Introduction

most detection methods are still constrained to a small set of objects

目标检测受限于小目标检测



labelling images for detection is far more expensive than labelling for classification or tagging 为检测标记图像比分类或标记要昂贵的多。



we propose a new method to harness the image amount of classification data we already have and use it to expand the scope of current detection systems.<u>**Our method uses a hierachical view of object classification that allows us to combine distinct datasets together**</u>使用对象分类的层次试图，允许我们将不同的数据集集合起来。



We also propose a <u>joint training algorithm</u>（連接訓練算法） that allows us to train object detectors on both detection and classification data. <u>**Our method leverages labeled detection images to learn to precisely localize objects while it uses classification images to increase its vocabulary and robustness.**</u>我们的方法利用標記的檢測圖像來學習精確定位對象，同時使用分類圖像來增加詞彙量和魯棒性





## Better

YOLO suffers from a variety of shortcomings relative to state-of-the-art detection systems. Error analysis of YOLO compared to Fast R-CNN shows that <u>YOLO makes a significant number of localization errors.</u> Furthermore, <u>YOLO has relatively low recall compared to region proposal-based methods.</u> Thus we focus mainly on improving recall and localization while maintaining classification accuracy

存在兩個問題

1. 定位錯誤多
2. 較低的召回率



Computer vision generally trends towards larger, deeper networks

<u>Instead of scaling up our network, we simplify the network and then make the representation easier to learn</u>與擴大網絡結構不同，我們讓特徵更容易學習。



### Batch Normalization批量歸一化

batch normalization leads to significant improvements in convergence while eliminating the need for other forms of regularization

批量歸一化顯著提高收斂性，同時消除了對其他形式的正則化的需要。



### High Resolution Classifier 高分辨率分類器

For YOLOv2 we first fine tune the classification network at the full 448 × 448 resolution for 10 epochs on ImageNet. <u>This gives the network time to adjust its filters to work better on higher resolution input.</u> We then fine tune the resulting network on detection. This high resolution classification network gives us an increase of almost 4% mAP.

擴大圖片尺寸，給網絡時間去調整過濾器在高分辨率圖片上，表現更好

### Convolutional With Anchor Boxes帶錨框的捲積

Instead of predicting coordinates directly Faster R-CNN predicts bounding boxes using **hand-picked priors** **精心挑選的先驗**



 Predicting offsets instead of coordinates simplifies the problem and makes it easier for the network to learn

預測偏差比直接預測坐標簡單，模型也好學習



We remove the fully connected layers from YOLO and use anchor boxes to predict bounding boxe

使用錨框預測邊界框





When we move to anchor boxes we also decouple the class prediction mechanism from the spatial location and instead predict class and objectness for every anchor box.

當我們移動錨框的時候，我們還將類別預測機制與空間位置分離，而不是為每個錨框預測類別和對象



### Dimension Clusters維度集群

We encounter two issues with anchor boxes when using them with YOLO. <u>The first is that the box dimensions are hand picked</u>. The network can learn to adjust the boxes appropriately but if we pick better priors for the network to start with we can make it easier for the network to learn to predict good detections

盒子的維度是先驗的。



Instead of choosing priors by hand, we run k-means clustering on the training set bounding boxes to automatically find good priors

我們不是手動選擇先驗，而是在訓練集邊界框上運行 k-means 聚類以自動找到好的先驗

 If we use standard k-means with Euclidean distance larger boxes generate more error than smaller boxes.如果我們使用標準的 k-means
歐氏距離較大的框比較小的框產生更多的錯誤。



We run k-means for various values of k and plot the average IOU with closest centroid, see Figure 2. We choose <u>k = 5</u>  as a good tradeoff between model complexity and high recall.

我們對各種 k 值運行 k-means 並繪製具有最接近質心的平均 IOU，參見圖 2。我們選擇 k = 5 作為模型複雜性和高召回率之間的良好折衷。





### Direct location prediction 直接定位預測

When using anchor boxes with YOLO we encounter a second issue: model instability, especially during early iterations. 

使用錨框時對於 YOLO，我們遇到了第二個問題：模型不穩定，特別是在早期迭代期間。



For example, a prediction of tx = 1 would shift the box to the right by the width of the anchor box, a prediction of tx = −1 would shift it to the left by the same amount.



<u>With random initialization the model takes a long time to stabilize to predicting sensible offsets.</u>

隨機初始化模型，花很長時間，去穩定地預測合理偏差

Instead of predicting offsets we follow the approach of YOLO and predict location coordinates relative to the location of the grid cell. This bounds the ground truth to fall between 0 and 1. <u>**We use a logistic activation to constrain the network’s predictions to fall in this range.**</u>

限制位置預測使得更穩定，易於學習。



### Fine-Grained Features 細粒度特徵

We take a different approach, simply adding a passthrough layer that brings features from an earlier layer at 26 × 26 resolution.

添加一個穿透層，以26x26分辨率從從前層中提取特徵。



The passthrough layer concatenates the higher resolution features with the low resolution features by stacking adjacent features into different channels instead of spatial locations, 穿透層通過將相鄰特徵堆疊到不同的通道，而不是空間位置，將高分辨率特徵與低分辨率特徵連結起來。



### Multi-Scale Training 多尺度訓練

We want YOLOv2 to be robust to running on images of different sizes so we train this into the model.

Instead of fixing the input image size we change the network every few iterations

每隔幾次迭代，我們將會改變圖片的輸入尺寸

This regime forces the network to learn to predict well across a variety of input dimensions

這種制度迫使網絡學會在各種輸入維度上進行良好預測



### Further Experiments 更多的實驗

....



## Faster

<u>Most detection frameworks rely on VGG-16 as the base feature extractor . VGG-16 is a powerful, accurate classification network but it is needlessly complex.</u> The convolutional layers of VGG-16 require 30.69 billion floating point operations for a single pass over a single image at 224 × 224 resolution.

大多數的目標檢測框架核心是VGG16，很準確但是也很複雜和慢



The YOLO framework uses a custom network based on the Googlenet architecture

YOLO使用GoogleNet作為參照，但是少了一些準確性



>We propose a new classification model to be used as the base of YOLOv2. Our model builds off of prior work on network design as well as common knowledge in the field. Similar to the VGG models we use mostly 3 × 3 filters and double the number of channels after every pooling step [17]. Following the work on Network in Network (NIN) we use global average pooling to make predictions as well as 1 × 1 filters to compress the feature representation between 3 × 3 convolutions [9]. We use batch normalization to stabilize training, speed up convergence, and regularize the model [7].

我們最後的模型叫做darknet-19，訓練參數進一步減少





### Training for classification為分類的訓練

We train the network on the standard ImageNet 1000 class classification dataset for 160 epochs using stochastic gradient descent with a starting learning rate of 0.1, polynomial rate decay with a power of 4, weight decay of 0.0005 and momentum of 0.9 using the Darknet neural network framework 

我們在標準 ImageNet 1000 類分類數據集上訓練網絡 160 個時期，使用隨機梯度下降，起始學習率為 0.1，多項式速率衰減為 4，權重衰減為 0.0005，動量為 0.9，使用 Darknet 神經網絡框架





## Stronger

We propose a mechanism for jointly training on classification and detection data. 

我們提出了一種聯合訓練分類和檢測數據的機制。



During training we mix images from both detection and classification datasets. <u>1⃣️When our network sees an im</u>age labelled for detection we can backpropagate based on the full YOLOv2 loss function. <u>2⃣️When it sees a classificatio</u>n image we only backpropagate loss from the classificationspecific parts of the architecture.

1⃣️當網絡看到一個標記為檢測的圖像，基於完整的yolov2損失函數進行反向傳播

2⃣️當網絡看到一個分類圖像，從架構的分類特定部分中反向傳播損失





This approach presents a few challenges. <u>Detection datasets have only common objects and general labels</u>, like “dog” or “boat”. Classification datasets have a much wider and deeper range of labels. ImageNet has more than a hundred breeds of dog



If we want to train on both datasets we need a coherent way to merge these labels.

使用連貫方法去merge類別



Most approaches to classification <u>use a softmax layer across all the possible categories to compute the fina</u>l probability distribution. Using a softmax assumes the classes are mutually exclusive

使用softmax是相互排斥的，



We could instead use a <u>**multi-label model**</u> to combine the datasets which does not assume mutual exclusion. This approach ignores all the structure we do know about the data, for example that all of the COCO classes are mutually exclusive

我們可以改為使用多標籤模型來組合
不假設互斥的數據集。 這種方法忽略了我們所知道的關於數據的所有結構，
例如，所有 COCO 類都是互斥的



### Hierarchical classification分級分類

we simplify the problem by building a hierarchical tree from the concepts in ImageNet.

在ImageNet中建立概念的分級樹

> To build this tree we examine the visual nouns in ImageNet and look at their paths through the WordNet graph to
> the root node, in this case “physical object”. Many synsets
> only have one path through the graph so first we add all of
> those paths to our tree. Then we iteratively examine the
> concepts we have left and add the paths that grow the tree
> by as little as possible. So if a concept has two paths to the
> root and one path would add three edges to our tree and the
> other would only add one edge, we choose the shorter path.
>
> 為了構建這棵樹，我們檢查了 ImageNet 中的視覺名詞並查看它們通過 WordNet 圖的路徑
> 根節點，在本例中為“物理對象”。 許多同義詞集
> 只有一條路徑通過圖表，所以首先我們添加所有
> 那些通往我們樹的路徑。 然後我們反複檢查
> 我們留下的概念並添加生長樹的路徑
> 盡可能少。 所以如果一個概念有兩條路徑
> 根和一條路徑會為我們的樹添加三個邊，而
> 其他只會添加一條邊，我們選擇較短的路徑。

 So if a concept has two paths to the root and one path would add three edges to our tree and the other would only add one edge, 找到最短路徑





The final result is WordTree, a hierarchical model of visual concepts. To perform classification with WordTree we predict conditional probabilities at every node for the probability of each hyponym of that synset given that synset. For example, at the “terrier” node we predict:最終的結果是 WordTree，一種視覺概念的層次模型。 為了使用 WordTree 進行分類，我們
在給定同義詞集的情況下，為該同義詞集的每個下位詞的概率預測每個節點的條件概率。 





. During training we propagate ground truth labels up the tree so that if an image is labelled as a “Norfolk terrier” it also gets labelled as a “dog” and a “mammal”, etc. . 在訓練中
我們在樹上傳播真值標籤，這樣如果一個圖像被標記為“諾福克梗”，它也會被標記為
“狗”和“哺乳動物”等





Performance degrades gracefully on new or unknown object categories. For example, if the network sees a picture of a dog but is uncertain what type of dog it is, it will still predict “dog” with high confidence but have lower confidences spread out among the hyponyms

在未知目標上表現良好





### Dataset combination with WordTree

We simply map the categories in the datasets to synsets in the tree. 



### Joint classification and detection







# 实现代码

https://www.maskaravivek.com/post/yolov2/
