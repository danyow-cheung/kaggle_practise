'''ref
1. https://towardsdatascience.com/xception-from-scratch-using-tensorflow-even-better-than-inception-940fb231ced9
2. https://stephan-osterburg.gitbook.io/coding/coding/ml-dl/tensorfow/ch3-xception/training-and-evaluating-xception-model
'''
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Conv2D,Add
from tensorflow.keras.layers import SeparableConv2D,ReLU
from tensorflow.keras.layers import BatchNormalization,MaxPool2D
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras import Model
import os 


'''creating the Conv-BatchNorm block'''
def conv_bn(x,filters,kernel_size,strides=1):
    x= Conv2D(filters=filters,kernel_size=kernel_size,padding='same',use_bias=False,strides=strides)(x)

    x = BatchNormalization()(x)
    return x 

'''creating the separableConv-BatchNorm block'''
def sep_bn(x,filters,kernel_size,strides=1):
    x = SeparableConv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding='same',use_bias=False)(x)

    x = BatchNormalization()(x)
    return x 


'''function for entry,middle and exit flow'''
def entry_flow(x):
    x = conv_bn(x,filters=32,strides=2,kernel_size=3)
    x = ReLU()(x)

    x = conv_bn(x,filters=64,kernel_size=3,strides=1)
    tensor = ReLU()(x)

    x = sep_bn(tensor,filters=128,kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x,filters=128,kernel_size=3)
    x = MaxPool2D(pool_size=3,strides=2,padding='same')(x)
    
    # 残差连接
    tensor = conv_bn(tensor,filters=128,kernel_size=1,strides=2)
    x = Add()([tensor,x])

    x= ReLU()(x)
    x = sep_bn(x,filters=256,kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x,filters=256,kernel_size=3)
    x = MaxPool2D(pool_size=3,strides=2,padding='same')(x)

    tensor = conv_bn(tensor,filters=256,kernel_size=1,strides=2)
    x = Add()([tensor,x])

    x = ReLU()(x)
    x = sep_bn(x,filters=728,kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x,filters=728,kernel_size=3)
    x = MaxPool2D(pool_size=3,strides=2)
    tensor = conv_bn(tensor,filters=758,kernel_size=1,strides=2)
    x = Add()([tensor,x])
    return x 

def entry_flow_v2(x):
    x = conv_bn(x, filters =32, kernel_size =3, strides=2)
    x = ReLU()(x)
    x = conv_bn(x, filters =64, kernel_size =3, strides=1)
    tensor = ReLU()(x)
    
    x = sep_bn(tensor, filters = 128, kernel_size =3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 128, kernel_size =3)
    x = MaxPool2D(pool_size=3, strides=2, padding = 'same')(x)
    
    tensor = conv_bn(tensor, filters=128, kernel_size = 1,strides=2)
    x = Add()([tensor,x])
    
    x = ReLU()(x)
    x = sep_bn(x, filters =256, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters =256, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding = 'same')(x)
    
    tensor = conv_bn(tensor, filters=256, kernel_size = 1,strides=2)
    x = Add()([tensor,x])
    
    x = ReLU()(x)
    x = sep_bn(x, filters =728, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters =728, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding = 'same')(x)
    
    tensor = conv_bn(tensor, filters=728, kernel_size = 1,strides=2)
    x = Add()([tensor,x])
    return x 


def middle_flow(tensor):
    # repeat 8 times 
    for _ in range(8):
        x = ReLU()(tensor)
        x = sep_bn(x,filters=758,kernel_size=3)
        x = ReLU()(x)
        x = sep_bn(x,filters=758,kernel_size=3)
        
        x = ReLU()(x)
        x = sep_bn(x,filters=758,kernel_size=3)
        x = ReLU()(x)
        tensor = Add()([tensor,x])
    return tensor 

def middle_flow_v2(tensor):
    for _ in range(8):
        x = ReLU()(tensor)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        tensor = Add()([tensor,x])
        
    return tensor

def exit_flow(tensor):
    x = ReLU(tensor)
    x =sep_bn(x,filters=728,kernel_size=3)
    x = ReLU()(x)
    x =sep_bn(x,filters=1024,kernel_size=3)
    x = MaxPool2D(pool_size=3,strides=2,padding='same')(x)

    tensor = conv_bn(tensor,filters=1024,kernel_size=1,strides=2)
    x = Add()([tensor,x])
    x = sep_bn(x,filters=1536,kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x,filters=2048,kernel_size=3)
    x = ReLU()(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(units=1000,activation='softmax')(x)
    return x 


def exit_flow_v2(tensor):
    x = ReLU()(tensor)
    x = sep_bn(x, filters = 728,  kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 1024,  kernel_size=3)
    x = MaxPool2D(pool_size = 3, strides = 2, padding ='same')(x)
    
    tensor = conv_bn(tensor, filters =1024, kernel_size=1, strides =2)
    x = Add()([tensor,x])
    
    x = sep_bn(x, filters = 1536,  kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 2048,  kernel_size=3)
    x = GlobalAvgPool2D()(x)
    
    x = Dense (units = 1000, activation = 'softmax')(x)
    
    return x

'''
creating the xception model
'''
def create_model():
    input = Input(shape=(259,259,3))
    x = entry_flow_v2(input)
    x = middle_flow_v2(x)
    output = exit_flow_v2(x)

    model = Model (inputs=input, outputs=output)
    model.summary()


    '''calculate trainable parameters'''
    import tensorflow.keras.backend as K 
    import numpy as np

    sum_parameters = np.sum([K.count_params(p) for p in model.trainanle_weights])
    print(sum_parameters)


'''Image Preprocessing And Augmentation'''
def preprocess_image(x):
    x /=255 
    x-=0.5 
    x*=0.2 
    # convert RGB to BGR
    x = x[...,::-1]
    # Zero-center by mean pixel
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    return x

'''train and evaluating Xception Model
'''
def train_xception_model():
    # load the pre-trained xception model 
    models = tf.contrib.keras.models
    layers = tf.contrib.keras.layers 
    uitls = tf.contrib.keras.utils 
    losses = tf.contrib.keras.losses 
    optimizers = tf.contrib.keras.optimizers 
    metrics = tf.contrib.keras.metrics 
    preprocessing_image = tf.contrib.keras.preprocessing.image 
    applications= tf.contrib.kersa.applications 
    

    base_model = applications.Xception(include_top=False,weights='imagenet',input_shape=(299,299,3),pooling='avg')
    # output for convolutional layers
    x = base_model.output 
    # final dense layer 
    outputs = layers.Dense(4,activation='softmax')(x)
    # define model with base_model 
    model = models.Model(inputs=base_model.input,outputs=outputs)
    
    # freeze weights of early layer to ease training 
    for layer in model.layers[:40]:
        layer.trainable=False
    
    loss = losses.categorical_crossentropy
    optimizer = optimizers.RMSprop(lr=0.0001)
    metric = [metrics.categorical_accuracy]
    # compile model
    model.compile(optimizer,loss,metric)
    model.summary()
    
    train_datagen = preprocessing_image.ImageDataGenerator(
                    preprocessing_function=preprocess_image,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)

    test_datagen = preprocessing_image.ImageDataGenerator(
                    preprocessing_function=preprocess_image)
    
    BASE_DIR = "/Users/marvinbertin/Github/marvin/ImageNet_Utils"

    train_generator = train_datagen.flow_from_directory(
        os.path.join(BASE_DIR, "imageNet_dataset/train"),
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical',
        shuffle=True)

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(BASE_DIR, "imageNet_dataset/validation"),
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical',
        shuffle=True)

    history = model.fit_generator(
        train_datagen,
        steps_per_epoch = 80,
        epochs=10,
        validation_data = validation_generator,
        validation_steps = 20
    )
        
    print(history.history['accuracy'])


if __name__ =='__main__':
    pass 
