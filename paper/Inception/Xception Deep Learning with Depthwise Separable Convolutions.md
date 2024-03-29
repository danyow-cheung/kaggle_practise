# Xception: Deep Learning with Depthwise Separable Convolutions

> **可分離捲積**分爲兩種
>
> 1. 空間可分離捲積
> 2. 深度可分離捲積
>
> **空間可分離捲積**
>
> it 主要處理圖像和内核的空間維度：寬和高（另一維度，’深度‘維度，是每個圖像的通道數）
>
> 解釋
>
> ```
> [3 6 9       [3 
>  4 8 12       4
>  5 10 15] =   5] x [1 2 3]
> ```
>
> 现在，我们不再进行一次 9 次乘法的卷积，而是进行两次 3 次乘法（总共 6 次）的卷积，以达到相同的效果。通过较少的乘法，计算复杂性会降低，并且网络能够运行得更快。
>
> 最著名的空間可分離捲積是Sobel核，用於檢測邊緣
>
> ```
> [-1  0  1     [1 
>  -2  0  2      2
>  -1  0  1] =   1] x [-1 0 1]
> ```
>
> **深度可分離捲積**
>
> it 處理空間維度，還能處理深度（通道數）。比如説，輸入圖像是3個通道。經過捲積可能有多個通道。
>
> 
>
> 深度可分離捲積分爲兩個部分，深度捲積和點捲積
>
> 【深度捲積】<img src = 'https://miro.medium.com/v2/resize:fit:720/format:webp/1*yG6z6ESzsRW-9q5F_neOsg.png'>
>
> 【駐點捲積】
>
> 请记住，原始卷积将 12x12x3 图像转换为 8x8x256 图像。目前，深度卷积已将 12x12x3 图像转换为 8x8x3 图像。现在，我们需要增加每个图像的通道数。
>
> 逐点卷积之所以如此命名，是因为它使用 1x1 内核，或者迭代每个点的内核。无论输入图像有多少个通道，该内核的深度都相同；在我们的例子中，为 3。因此，我们通过 8x8x3 图像迭代 1x1x3 内核，以获得 8x8x1 图像。
>
> <img src ='https://miro.medium.com/v2/resize:fit:720/format:webp/1*37sVdBZZ9VK50pcAklh8AQ.png'>



## Introduction

捲積神經網絡都是，一堆捲積神經元的堆來特徵提取和用於空間子採樣的最大池化操作。



## Prior work



## The Xception architecture 

