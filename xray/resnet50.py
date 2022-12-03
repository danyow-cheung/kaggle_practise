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

'''真正的残差块'''
def block1(x,filters,kernel_size=3,stride=1,conv_shortcut=True,name=None):
    '''

    :param x:
    :param filters:
    :param kernel_size:
    :param stride:
    :param conv_shortcut:
    :param name:
    :return:
    '''

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
        x = blocks1(
            x,filters,conv_shortcut =False,name=name+'_block'+str(i)
        )
    return x

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