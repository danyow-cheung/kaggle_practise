# VGG16论文记录和代码复现



## Abstract

研究网络深度在大尺度图片识别中的影响。

使用3x3卷积操作可以在现有技术配置，在16层至19层有较好的提升。



## CONVENT CONFIGURATIONS

### ARCHITECTURE

输入ConvNets是224x224的rgb图片，

> in one of the configurations we also utilise 1x1 convolution filters,which can be seen as a linear transformation of the input channels.(followed by non-linearity)

在其中一種配置中，我們還利用了 1x1 卷積濾波器，可以將其視為輸入通道的線性變換。（隨後是非線性）



模型中没有Local Respons Normalization



### DISCUSSION

比起使用较大的感受野在第一个卷积操作，

我們在整個網絡中使用非常小的 3x3 感受野。 



**使用3x3卷积而不是7x7卷积的原因是**



1. we incorporate three non-linearity rectfication layers instead of a single one.which makes the decision function more discriminative.
2. we decrease the number of parameters:assuming that both the input and the output for a weights....which can be seen as imposing a regularisatin on the 7x7 conv filters.forcing them to have a decomposition through the 3x3 filters(with non-linearity injected in between)

1. 我們合併了三個非線性校正層而不是一個。這使得決策函數更具辨別力。
2. .我們減少參數的數量：假設輸入和輸出都為權重....這可以看作是在 7x7 conv 過濾器上施加正則化。強制它們通過 3x3 過濾器進行分解（在兩者之間注入非線性）



**the incorporation of 1x1 conv layers**

1x1 conv layers 是一種在不影響 conv 接受域的情況下增加決策函數非線性的方法



## CLASSIFICATION FRAMEWORK

### TRAINING

多层逻辑斯蒂回归，批量的梯度下降。



网络权重的初始化也很重要。



使用各向同性缩放训练图像。

尺度抖动。

# 实现代码
<img src= 'https://raw.githubusercontent.com/blurred-machine/Data-Science/master/Deep%20Learning%20SOTA/img/network.png'>

```python
import keras 
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Conv2D,Dense,MaxPool2D,Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

'''load data'''
tr_data = ImageDataGenerator()
tr_data.flow_from_directory(directory='data',target_size=(244,244))
ts_data = ImageDataGenerator()
ts_data.flow_from_directory(directory='test',target_size=(244,244))

'''set up the model'''
model = Sequential()
model.add(Conv2D(input_shape=(244,244,3),filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=2))

model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=2))

model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=2))


model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=2))

model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=2))

# final output layer
model.add(Flatten())
model.add(Dense(units=4096,activation='relu'))
model.add(Dense(units=4096,activation='relu'))
model.add(Dense(units=2,activation='softmax'))

'''compile and train '''
from keras.optimizer_v1 import Adam
opt = Adam(lr=0.001)
model.compile(optimizer=opt,loss=keras.losses.categorical_crossentry,metrics=['accuracy'])
model.summary()



from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
hist = model.fit_generator(steps_per_epoch=100,generator=tr_data, validation_data= ts_data, validation_steps=10,epochs=100,callbacks=[checkpoint,early])
import matplotlib.pyplot as plt
plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()

```




# 参考

- [VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/pdf/1409.1556.pdf)
- https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
