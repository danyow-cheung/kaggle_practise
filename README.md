# kaggle_practise
 kaggle上練習搭建模型代碼


> [如何將kaggle導入到colab](https://www.kaggle.com/general/74235)
>
> 1. Go to your account, Scroll to API section and Click **Expire API Token** to remove previous tokens
>
> 2. Click on **Create New API Token** - It will download kaggle.json file on your machine.
>
> 3. Go to your Google Colab project file and run the following commands:
>
>    **1) ! pip install -q kaggle**
>
>    **2) from google.colab import files**
>
> **files.upload()**
>
> - Choose the kaggle.json file that you downloaded
>
> **3) ! mkdir ~/.kaggle**
>
> **! cp kaggle.json ~/.kaggle/**
>
> 
>
> 
>
> - Make directory named kaggle and copy kaggle.json file there.
>
> **4) ! chmod 600 ~/.kaggle/kaggle.json**
>
> - Change the permissions of the file.
>
> **5) ! kaggle datasets list**
> \- That's all ! You can check if everything's okay by running this command.
>
> ## Download Data
>
> **! kaggle competitions download -c 'name-of-competition'**
>
> Use unzip command to **unzip the data**:
>
> For example,
>
> Create a directory named train,
>
> **! mkdir train**
>
> unzip train data there,
>
> **! unzip train.zip -d train**




## Predict data(ml)

- [泰坦尼克乘客存活預測](https://github.com/danyow-cheung/kaggle_practise/tree/main/titanic)

- [石油洩漏區域預測](https://github.com/danyow-cheung/kaggle_practise/tree/main/oil_split)

- [泰坦尼克號宇宙飛船](https://github.com/danyow-cheung/kaggle_practise/tree/main/Spaceship-titanic)

  



## Predict data(dl)

- [股票预测数据集](https://github.com/danyow-cheung/kaggle_practise/tree/main/stock)

- [手写字体识别](https://github.com/danyow-cheung/kaggle_practise/tree/main/digits)

- [手語字母識別](https://github.com/danyow-cheung/kaggle_practise/tree/main/ASL)

  >.csv格式識別

- [GAN繪圖](https://github.com/danyow-cheung/kaggle_practise/tree/main/gan_arts)

  

## Image Classification 

- [胸部X光肺炎圖像](https://github.com/danyow-cheung/kaggle_practise/tree/main/xray)

- [速食食物图片分类](https://github.com/danyow-cheung/kaggle_practise/tree/main/fast%20food)

- [鞋子图片分类](https://github.com/danyow-cheung/kaggle_practise/tree/main/shoe)

- [工件缺陷檢測](https://github.com/danyow-cheung/kaggle_practise/tree/main/SteelDefect)

  



## Image Recognition

- [地形图片分类](https://github.com/danyow-cheung/kaggle_practise/tree/main/landscape)





### 练习目标

> 使用[keras.applications](https://keras.io/api/applications/)上的经典模型多跑分类等数据集

**按照模型出现的先后时间顺序进行学习**

<img src='paper/src/cnn_history.png'>



- [Xception✅](https://github.com/danyow-cheung/kaggle_practise/tree/main/paper/Xception)
- [AlexNet✅](https://github.com/danyow-cheung/kaggle_practise/blob/main/paper/AlexNet)
- [VGG16✅](https://github.com/danyow-cheung/kaggle_practise/blob/main/paper/VGG16)
- ~~VGG19~~
- [ResNet50✅]((https://github.com/danyow-cheung/kaggle_practise/blob/main/paper/ResNet))
- [ResNet50v2](((https://github.com/danyow-cheung/kaggle_practise/blob/main/paper/ResNet))
- ~~ResNet101~~ 
- ~~ResNet152~~ 
- ~~ResNet152V2~~
- [InceptionV3](https://github.com/danyow-cheung/kaggle_practise/blob/main/paper/Inception)
- InceptionResNetV2
- [MobileNet]((https://github.com/danyow-cheung/kaggle_practise/blob/main/paper/MobileNet))
- MobileNet
- [YOLO](https://github.com/danyow-cheung/kaggle_practise/blob/main/paper/YOLO)




todo：
[Image Matching Challenge 2022](https://www.kaggle.com/competitions/image-matching-challenge-2022/code)



