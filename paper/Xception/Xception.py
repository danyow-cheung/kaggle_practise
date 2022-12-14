'''ref
https://towardsdatascience.com/xception-from-scratch-using-tensorflow-even-better-than-inception-940fb231ced9
'''
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Conv2D,Add
from tensorflow.keras.layers import SeparableConv2D,ReLU
from tensorflow.keras.layers import BatchNormalization,MaxPool2D
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras import Model


'''creating the Conv-BatchNorm block'''
def conv_bn(x,filters,kernel_size,strides=1):
    x= Conv2D(filters=filters,kernel_size=kernel_size,padding='same',use_bias=False)(x)

    x = BatchNormalization()(x)
    return x 

'''creating the separableConv-BatchNorm block'''
def sep_bn(x,filters,kernel_size,strides=1):
    x = SeparableConv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding='same',use_bias=False)(x)

    x = BatchNormalization()(x)
    return x 


'''function for entry,middle and exit flow'''
def entry_flow():
    x = conv_bn(x,filters=32,strides=2,kernel_size=3)
    x = ReLU()(x)

    x = conv_bn(x,filters=64,kernel_size=3)
    tensor = ReLU()(x)

    x = sep_bn(tensor,filters=128,kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(filters=128,kernel_size=3)
    x = MaxPool2D(pool_size=3,strides=2)(x)
    
    # 残差连接
    tensor = conv_bn(tensor,filters=128,kernel_size=1,strides=2)
    x = Add()([tensor,x])

    x= ReLU()(x)
    x = sep_bn(x,filters=256,kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x,filters=256,kernel_size=3)
    x = MaxPool2D(pool_size=3,strides=2)(x)

    tensor = conv_bn(tensor,filters=256,kernel_size=1,strides=2)
    x = Add([tensor,x])

    x = ReLU()(x)
    x = sep_bn(x,filters=728,kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x,filters=728,kernel_size=3)
    x = MaxPool2D(pool_size=3,strides=2)
    tensor = conv_bn(tensor,filters=758,kernel_size=1,strides=2)
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


def exit_flow(tensor):
    x = ReLU(tensor)
    x =sep_bn(x,filters=728,kernel_size=3)
    x = ReLU()(x)
    x =sep_bn(x,filters=1024,kernel_size=3)
    x = MaxPool2D(pool_size=3,strides=2)
    tensor = conv_bn(tensor,filters=1024,kernel_size=1,strides=2)
    x = Add()([tensor,x])
    x = sep_bn(x,filters=1536,kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x,filters=2048,kernel_size=3)
    x = ReLU()(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(units=1000,activation='softmax')(x)
    return x 