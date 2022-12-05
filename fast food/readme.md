# Fast Food Classification


> 从评论区学的模型构建

## [ResNet50v2](https://www.kaggle.com/code/utkarshsaxenadn/fast-food-classification-resnet50v2-acc-92)

```python
from keras.layers import Dense,GlobalAveragePooling2D as GAP,Dropout

n_classes = 5 
'''main model'''
base = ResNet50V2(include_top=False,input_shape=(256,256,3))
base.trainable=False

model = Sequential([
  base,
  GAP,  Dense(1024,kernel_initializer='he_normal',activation='relu'),
  Dropout(0.4),
  Dense(n_classes,activation='softmax')
]) 

#callbacks
callback = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint('food-classifier.h5',
                    save_best_only=True)
]

model.compile(
  loss=''
	optimizer = tf.keras.optimizers.Adam(learning_rate=2e-3),
	metrics =['accuracy']
)

model.fit(train_ds,validation_data=val_ds,epochs=100,callbacks=callback)
```



## [BiT](https://www.kaggle.com/code/utkarshsaxenadn/fast-food-classification-bit-acc-94)

从架构角度来看，BiT不过是ResNet152V2的4倍缩放版本。这里的主要思想是转移学习，该模型在大数据集上进行了预训练，因此可以在子数据集或基本上其他小数据集上训练，由于该模型在非常大的数据集上预训练，预计它在小数据集上将表现出色。BiT有3种变体：

- BiT-L：这是在300M样本的图像分类任务上训练的（这是私人的）
- BiT-M：这是在14M样本的图像分类任务中训练的。
- BiT-S：这是在1.3M样本的图像分类任务上训练的。

```python
# load model by url 
url = "https://tfhub.dev/google/bit/m-r50x1/1"
from tensorflow_hub import KerasLayer as KL
bit = KL(url)
model_name = 'model_bit'
'''build the model'''
model = Sequential([
  InputLayer(input_shape=(256,256,3)),
  bit,
  Dropout(0.2),
  Dense(n_classes,activation='softmax',kernel_initializer='zeros')
],name=model_name)

'''compile and train model '''
lr = 5e-3 
lr_scheduler = PwCD(boundaries=[200,300,400],values=[lr*0.1, lr*0.01, lr*0.001, lr*0.0001])

opt = SGD(learning_rate =lr_scheduler,momentum=0.9)
model.compile(
	loss='sparse_categorical_crossentropy',
	optimizer =opt,
	metrics=['accuracy'])

# Callbacks
cbs = [ES(patience=5, restore_best_weights=True), MC(model_name+".h5", save_best_only=True)]

# Training
history = model.fit(train_ds, validation_data=valid_ds, epochs=50, callbacks=cbs)
```



## 本人东凑西凑傻瓜模型

> ResNet50

```python
from tensorflow.keras.applications.resnet50 import ResNet50
'''model head '''
model = ResNet50(weights='imagenet',include_top=False)
# from keras.layers.serialization import activation
result = model.output 
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
result = GlobalAveragePooling2D()(result)
result = Dense(512,activation='relu')(result)
predictions = Dense(5,activation='sigmoid')(result)

import tensorflow as tf 
from tensorflow.keras.models import Model
'''set up the model '''
resnet_model = Model(inputs=model.input,outputs=predictions)
'''compile model'''
resnet_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy']
)

'''model train '''
with tf.device("/device:GPU:0"):    
    history = resnet_model.fit_generator(
        train_generator,
        epochs=50,
        shuffle=True,
        verbose=1,
        validation_data=val_generator)
    
```

