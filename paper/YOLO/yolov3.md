# An Inctemental Improvement



## The Deal

### Bounding Box Prediction邊界框預測

在訓練期間，我們使用誤差平方和損失。 如果
一些坐標預測的基本事實是 `t^*`
我們的梯度是地面真實值（從地面計算
真值框）減去我們的預測：`t^* - t*`

通過反轉上面的等式可以很容易地計算出這個真實值。



### Class Prediction類別預測

Each box predicts the classes the bounding box may contain using multilabel classification. We do not use a softmax as we have found it is unnecessary for good performance, instead we simply use independent logistic classifiers. <u>During training we use binary cross-entropy loss for the class predictions.</u>

每個框使用多標籤分類預測邊界框可能包含的類。 我們不使用 softmax，因為我們發現它對於良好的性能是不必要的，<u>而是我們簡單地使用獨立的邏輯分類器。 在訓練期間，我們使用二元交叉熵損失進行類別預測</u>。



### Predictions Across Scales跨尺度預測

YOLOv3 <u>predicts boxes at 3 different scales,</u>Our systeam extracts features from those scales using a similar concept to feature pyramid networks

YOLOv3 <u>預測 3 種不同尺度的框</u>，我們的系統使用與特徵金字塔網絡類似的概念從這些尺度中提取特徵。

From our base feature extractor we add several convolutional layers. The last of these predicts a 3-d tensor encoding bounding box, objectness, and class predictions.

從我們的基本特徵提取器中，我們添加了幾個卷積層。 最後一個預測 3-d 張量編碼邊界框、目標性和類別預測。





### Feature Extractor 特徵提取

我們使用一個新的網絡來執行特徵提取。
我們的新網絡是 YOLOv2、Darknet-19 中使用的網絡和新奇的殘差網絡材料之間的混合方法。





