import  tensorflow as tf
from  tensorflow import keras
import  matplotlib.pyplot as plt
import  os
import time
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import  Conv2D,Dense,BatchNormalization,MaxPooling2D,Flatten,Dropout

'''Dataset:CIFAR-10 dataset'''
# load dataset
(train_images,train_labels),(test_images,test_labels) =cifar10.load_data()
# label names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# set up the train/val datasets (a small version)
validation_images,validation_labels = train_images[:5000],train_labels[:5000]
train_images,train_labels = train_images[5000:],train_labels[5000:]

test_images,test_labels = train_labels[5000:10000],train_labels[5000:10000]# add for model evaluation

# convert image to np.array
train_ds = tf.data.Dataset.from_tensor_slices((train_images,train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((validation_images,validation_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images,test_labels))# add for model evaluation

'''preporcessing image--data augmentation'''
def process_images(image,label):
    # normalize images to have a mean of 0 and standred deviation of 1
    image = tf.image.per_image_standardization(image)
    # resize
    image = tf.image.resize(image,(256,256))
    return image,label

'''Input Pipeline'''
train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
val_ds_size = tf.data.experimental.cardinality(val_ds).numpy()

train_ds = (train_ds.map(process_images)
                    .shuffle(buffer_size=train_ds_size)
                    .batch(batch_size=32,drop_remainder=True))
val_ds = (val_ds.map(process_images)
                .shuffle(buffer_size=train_ds_size)
                .batch(batch_size=32,drop_remainder=True))

# add for model evaluation
test_ds = (test_ds.map(process_images)
                .shuffle(buffer_size=train_ds_size)
                .batch(batch_size=32,drop_remainder=True))

'''AlexNet model'''
model = Sequential()
model.add(Conv2D(filters=96,kernel_size=11,strides=4,activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(3,3),strides=1))

model.add(Conv2D(filters=256,kernel_size=5,strides=1,activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(3,3),strides=1))

model.add(Conv2D(filters=384,kernel_size=3,strides=1,activation='relu',padding='same'))

model.add(Conv2D(filters=256,kernel_size=3,strides=1,activation='relu',padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())

model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(CLASS_NAMES),activation='softmax'))
model.compile(optimizer='sgd',loss='mse',metrics=['accuracy'])
model.summary()

history = model.fit(train_ds,
          epochs=50,
          validation_data=val_ds,
          validation_freq=1,
          )

# summarize history for accuracy
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Pretrained'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Pretrained'], loc='upper left')
plt.show()
# history accuracy
print(history.history['accuracy'][-1])

# model evaluation
model.evaluate(test_ds)