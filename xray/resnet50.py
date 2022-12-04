'''
tf.keras.applications.ResNet50 source code
查看resnet50 源码
https://github.com/keras-team/keras/blob/v2.11.0/keras/applications/resnet.py#L499-L533

'''

import tensorflow as tf

from keras import backend
from keras.applications import imagenet_utils
from keras.engine import training
from keras.layers import VersionAwareLayers
from keras.utils import data_utils
from keras.utils import layer_utils

layers = None

'''真正的残差块'''
def block1(x,filters,kernel_size=3,stride=1,conv_shortcut=True,name=None):
    '''

    :param x:输入张量
    :param filters:瓶颈层的过滤器数量
    :param kernel_size:
    :param stride:
    :param conv_shortcut:
    :param name:
    :return:
    '''
    bn_axis = 3 if backend.image_data_format()=='channels_last'else 1
    if conv_shortcut:
        shorcut = layers.Conv2D(
            4*filters,1,strides=stride,name=name+'_0_conv'

        )(x)
        shorcut = layers.BatchNormalization(
            axis=bn_axis,epsilon=1.001e-5,name=name+'_0_bn'
        )(shorcut)
    else:
        shortcut = x
    x = layers.Conv2D(filters,1,strides=stride,name=name+'_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,epsilon=1.001e-5,name=name+'_1_bn'
    )(x)
    x = layers.Activation('relu',name=name+'_1_relu')(x)
    x = layers.Conv2D(
        filters,kernel_size,padding='SAME',name=name+'_2_conv'
    )(x)

    x = layers.BatchNormalization(
        axis=bn_axis,epsilon=1.001e-5,name=name+'_2_bn'
    )(x)

    x = layers.Activation('relu',name=name+'_2_relu')(x)

    x = layers.Conv2D(4*filters,1,name=name+'_3_conv')(x)

    x = layers.BatchNormalization(
        axis = bn_axis,epsilon=1.001e-5,name=name+'_3_bn'
    )(x)

    x = layers.Add(name=name+'_add')([shortcut,x])
    x = layers.Activation('relu',name=name+'_out')(x)
    return x

'''残差块的组成集合'''
def stack1(x,filters,blocks,stride1 = 2,name=None):
    '''

    :param x:输入张量
    :param filters: 在代码块中瓶颈层的过滤器
    :param blocks: 多少个残差块
    :param stride1: 在首层网络的步长
    :param name:标签
    :return:残差块的张量
    '''
    x = block1(x,filters,stride=stride1,name=name+'_block1')
    for i in range(2,blocks+1):
        x = block1(
            x,filters,conv_shortcut =False,name=name+'_block'+str(i)
        )
    return x

def ResNet(
        stack_fn,
        preact,
        use_bias,
        model_name='resnet',
        include_top = True,
        weights = 'imagenet',
        input_tensor = None,
        input_shape = None,
        pooling=None,
        classes = 1000,
        classifier_activation='softmax',
        **kwargs
):
    '''
        实例化ResNet、ResNetV2和ResNeXt体系结构
        :param stack_fn:
        :param preact:
        :param use_bias:
        :param model_name:
        :param include_top:
        :param weights:
        :param input_tensor:
        :param input_shape:
        :param pooling:
        :param classes:
        :param classifier_activation:
        :param kwargs:
        :return:
        '''

    global layers
    if "layers" in kwargs:
        layers = kwargs.pop("layers")
    else:
        layers = VersionAwareLayers()
    if kwargs:
        raise ValueError(f'Unknown arguments(s):{kwargs}')
    '''...适当行忽略了部分的内容'''
    # 定义正确的输入维度
    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor,shape=input_shape)
        else:
            img_input = input_tensor


    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    x = layers.ZeroPadding2D(padding=((3,3),(3,3)),name='conv1_pad')(img_input )
    model = training.Model(inputs,x,name=model_name)

    return model


'''构建ResNet的api'''
def ResNet50(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape = None,
        pooling= None,
        classes = 1000,
        **kwargs,
        ):
    '''初始化ResNet网络结构'''
    def stack_fn(x):
        x = stack1(x,64,3,stride=1,name='conv2')
        x = stack1(x,128,4,name='conv3')
        x = stack1(x,256,6,name='conv4')
        return stack1(x,512,3,name='conv5')

    return ResNet(
        stack_fn,
        False,
        True,
        'resnet50',
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs,
    )