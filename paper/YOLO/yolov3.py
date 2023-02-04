'''
1. https://pylessons.com/YOLOv3-TF2-introduction
2. https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe
'''

'''DarkNet-53  from web1'''
def DarkNet53(input_data):
	input_data = convolutional(input_data,(3,3,3,32))
	input_data = convolutional(input_data,(3,3,32,64),downsample=True)

	for i in range(1):
		input_data = residual_block(input_data,64,32,64)

	
	input_data = convolutional(input_data,(3,3,64,128),downsample=True)

	for i in range(2):
		input_data = residual_block(input_data,128,64,128)

	input_data = convolutional(input_data,(3,3,128,256),downsample=True)

	for i in range(8):
		input_data = convolutional(input_data,512,256,512)

	route_1 = input_data
	input_data = convolutional(input_data,(3,3,256,512),downsample=True)
	for i in range(8):
		input_data = convolutional(input_data,512,256,512)

	route_2 = input_data
	input_data = convolutional(input_data,(3,3,512,1024),downsample=True)

	for i in range(4):
		input_data = convolutional(input_data,1024,512,1024)
	return route_1,route_2,input_data

'''
DarkNet53 from web2
code repo:https://github.com/mystic123/tensorflow-yolo-v3
'''

'''initialize the model and load pre-trained weights'''
import tensorflow as tf 

slim = tf.contrib.slim

def darknet53(inputs):
	'''
	build the DarkNet53 model
	'''
	inputs = _conv2d_fixed_padding(inputs,32,3)
	inputs = _conv2d_fixed_padding(inputs,64,3,strides=2)
	inputs = _darknet53_block(inputs,32)
	inputs = _conv2d_fixed_padding(inputs,128,3,strides=2)

	for i in range(2):
		inputs = _darknet53_block(inputs,54)
	inputs = _conv2d_fixed_padding(inputs,256,3,strides=2)

	for i in range(8):
		inputs = _darknet53_block(inputs,128)
	route_1 = inputs 
	inputs = _conv2d_fixed_padding(inputs,512,3,strides=2)

	for i in range(8):
		inputs = _darknet53_block(inputs,256)
	route_2 = inputs 
	inputs = _conv2d_fixed_padding(inputs,1024,3,strides=2)

	for i in range(4):
		inputs = _darknet53_block(inputs,512)

	return route_1,route_2,inputs 

def yolo_v3(inputs,num_classes,is_training=False,data_format='NCHW',reuse=False):
	'''
	Creates YOLOv3 model 
	:params inputs: 		a 4-d tensor of size [batch_size,height,width,channels].
							Demension batch_size may be underfined 
	:params num_classses:	number of predicted classes 
	:params is_training:	whether is training or not 
	:params data_format:	data format NCHW or NHWC
	:params reuse:			whether or not the network and its variables should be reused 
	:return : 
	'''
	with tf.variable_scope('darkent-53'):
		route_1,route_2,inputs = darknet53(inputs)
	with tf.variable_scope('yolo_v3'):
		route,inputs = _yolo_block(inputs,512)
		detect_1 = _detection_layer(inputs,num_classes,_ANCHORS[6:9],img_size,data_format)
		inputs = _conv2d_fixed_padding(route,256,1)
		upsample_size = route_2.get_shape().as_list()
		inputs = _upsample(inputs,upsample_size,data_format)
		inputs = tf.concat([inputs,route_2],axis=1 if data_format=='NCHW' else 3 )
		route,inputs = _yolo_block(inputs,256)

		detect_2 = _detection_layer(inputs,num_classes,_ANCHORS[3:6],img_size,data_format)
		detect_2 = tf.identity(detect_2,name='detect_2')
		inputs = _conv2d_fixed_padding(route,128,1)
		upsample_size = route_1.get_shape().as_list()
		inputs = _upsample(inputs,upsample_size,data_format)
		inputs = tf.concat([inputs,route_1],axis=1 if data_format=="NCHW" else 3)
		_,inputs = _yolo_block(inputs,128)
		detect_3 = _detection_layer(inputs,num_classes,_ANCHORS[0:3],img_size,data_format)
		detect_3 = tf.identity(detect_3,name='detect_3')
		detections = tf.concat([detect_1,detect_2,detect_3],axis=1)
		return detections



def load_weights(var_list,weights_file):
	'''
	Load and convert pre-trianed weights
	:param var_list:			list of network variables
	:param weights_file:		name of the binary file 
	:return : 
	'''
	with open(weights_file,'rb') as fp:
		_ = np.formfile(fp,dtype=np.int32,count=5)
		weights = np.formfile(fp,dtype=np.float32)

	'''
	Then we will use two pointers, 
	first to iterate over the list of variables var_list 
	and second to iterate over the list with loaded variables weights. 

	We need to check the type of the layer 
	following the one currently processed 
	and read appriopriate number of values. 
	In the code i will be iterating over 
	var_list and ptr will be iterating 
	over weights. We will return a list 
	of tf.assign ops. 
	I check the type of the layer simply by comparing it’s name. 
	'''
	ptr = 0 
	i = 0 
	assign_ops = []
	while i<len(var_list):
		var1 = var_list[i]
		var2 = var_list[i+1]
		# 

_BATCH_NORM_DECAY = 0.9 
_BATCH_NORM_EPSILON =1e-5 
_LEAKY_RELU = 0.1 
_ANCHORS  = [(10,13),(16,30),(33,23),(30,61),(62,45),(59,119),(116,90),(156,198),(373,326)]

# transpose the inputs to NCHW 
if data_format == "NCHW":
	inputs = tf.transpose(inputs,[0,3,1,2])

# normalize values to range[0..1]
inputs = inputs/255 

# set batch norm params 
batch_norm_params = {
	'decay':_BATCH_NORM_DECAY,
	'epsilon':_BATCH_NORM_EPSILON,
	'scale':True,
	'is_training':is_training,
	"fused":None , # use fused batch norm if possible
}

# set activatioin_fn and parameters for conv2d ,batch_norm
with slim.arg_scope([slim.conv2d,slim.batch_norm,_fixed_padding],data_format=data_format,reuse=True):
	with slim.arg_scope([slim.conv2d],normalizer_fn = slim.batch_norm,normalizer_params = batch_norm_params,biases_initializer=None,activation_fn = lambda x:tf.nn.leaky_relu(x,alpha=_LEAKY_RELU)):
		with tf.variable_scope('darkent-53'):
			inputs = darknet53(inputs)

'''
Before we define convolutional layers, 
we have to realize that authors’ implementation 
uses fixed padding independently of input size.
'''
@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs,kernel_size,*args,mode='CONSTANT',**kwargs):
	'''
	Pads the input along the spatial dimensions independently of input size 

	Args:
		inputs:     	a tensor of size [batch,channels,height_in,width_in] or 
				    	[batch,height_in,width_in,channels] depending on data_format 
		kernel_size:	The kernel to be used in the conv2d or max_pool2d operation should be a postive integer
		data_format: 	The input format('NHWC'or 'NCHW')
		mode:			The mode for tf.pad 
	Returns:
		A tensor with the same format as the input with the data either intact (if kernel_size==1) or padded(if kernel_size>1)

	'''
	pad_total = kernel_size -1 
	pad_beg = pad_total//2 
	pad_end = pad_total - pad_beg
	if kwargs['data_format']=='NCHW':
		padded_inputs = tf.pad(inputs,[[0,0],[0,0],[pad_beg,pad_end],[pad_beg,pad_end]],mode=mode)
	else:
		padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],[pad_beg, pad_end], [0, 0]], mode=mode)
		return padded_inputs

def _conv2d_fixed_padding(inputs,filters,kernel_size,strides=1):
	if strides>1:
		inputs = _fixed_padding(inputs,kernel_size)
	inputs = slim.conv2d(inputs,filters,kernel_size,strides=strides,padding=('SAME' if strides==1 else'VALID'))
	return inputs 

'''
Darknet-53模型是由一些具有2个凸层的块和快捷连接以及下采样层构建的。为了避免样板代码，我们定义了_darknet_block函数：
'''
def _darknet53_block(inputs,filters):
	shortcut = inputs 
	inputs = _conv2d_fixed_padding(inputs,filters,1)
	inputs = _conv2d_fixed_padding(inputs,filters*2,3)
	inputs = inputs+shortcut
	return inputs 


def _yolo_block(inputs,filters):
	inputs = _conv2d_fixed_padding(inputs,filters,1)
	inputs = _conv2d_fixed_padding(inputs,filters*2,3)
	
	inputs = _conv2d_fixed_padding(inputs,filters,1)
	inputs = _conv2d_fixed_padding(inputs,filters*2,3)
	inputs = _conv2d_fixed_padding(inputs,filters,1)
	route = inputs 
	inputs = _conv2d_fixed_padding(inputs,filters*2,3)
	return route,inputs 

def _detection_layer(inputs,num_classes,anchors,img_size,data_format):
	num_anchors = len(anchors)
	predictions = slim.conv2d(
		inputs,
		num_anchors*(5+num_classes),1,
		stride=1,
		normalizer_fn=None,
		activation_fn=None,
		biases_initializer=tf.zeros_initializer()
	)

	shape = predictions.get_shape().as_list()
	grid_size = _get_size(shape,data_format)
	dim = grid_size[0]*grid_size[1]
	bbox_atts = 5 + num_classes

	if data_format=='NCHW':
		predictions = tf.reshape(predictions,[-1,num_anchors*bbox_atts,dim])
		predictions = tf.transpose(predictions,[0,2,1])

	predictions = tf.reshape(predictions,[-1,num_anchors*dim,bbox_atts])

	stride = (img_size[0]//grid_size[0],img_size[1]//grid_size[1])
	anchors = [(a[0]/stride[0],a[1]/stride[1]) for a in anchors]

	box_centers,box_sizes,confidence,classes = tf.split(predictions,[2,2,1,num_classes],axis=-1)
	box_centers = tf.nn.sigmoid(box_centers)
	confidence = tf.nn.sigmoid(confidence)

	grid_x = tf.range(grid_size[0],dtype=tf.float32)
	grid_y = tf.range(grid_size[1],dtype=tf.float32)

	x_offset = tf.reshape(a,(-1,-1))
	y_offset = tf.reshape(b,(-1,-1))

	x_y_offset = tf.concat([x_offset,y_offset],axis=-1)
	x_y_offset = tf.reshape(tf.tile(x_y_offset,[1,num_anchors]),[1,-1,2])

	box_centers = box_centers+x_y_offset
	box_centers =box_centers*stride 

	anchors = tf.tile(anchors,[dim,1])
	box_sizes = tf.exp(box_sizes)*anchors 
	box_size = box_sizes*stride

	detections = tf.concat([box_centers,box_sizes,confidence],axis=-1)
	classes = tf.nn.sigmoid(classes)
	predictions = tf.concat([detections,classes],axis=-1)
	return predictions

def _get_size(shape,data_format):
	if len(shape)==4:
		shape = shape[1:]
	return shape[1:3] if data_format=='NCHW' else shape[0:2]
inputs = _fixed_padding(inputs,3,'NHWC',mode='SYMMETRIC')
'''这里有问题，原文blog'''
def _sample(inputs,out_shape,data_format='NCHW'):
	# we need to pad with one pixel so we set kernel_size=3 
	inputs = _fixed_padding(inputs,3,mode='SYMMETRIC')

	if data_format=="NCHW":
		inputs = tf.transpose(inputs,[0,2,3,1])
		height = out_shape[3]
		width = out_shape[2]
	else:
		height = out_shape[2]
		width = out_shape[1]

	# we padded with 1 pixel from each side and unsample by factor of 2 
	# so mew dimensions will be greater by 4 pixels after interpolation
	new_height = height + 4 
	new_width = width + 4 

	inputs = tf.image.resize_bilinear(inputs,(new_height,new_width))
	# trim back to desired size 
	inputs = inputs[:,2:-2,2:-2,:]

	# back to NCHW if needed 
	if data_format=='NCHW':
		inputs = tf.transpose(inputs,[0,3,1,2])
	inputs = tf.identity(inputs,name='unsamples')
	return inputs 

def _upsample(inputs,out_shape,data_format='NCHW'):
	if data_format=="NCHW":
		inputs = tf.transpose(inputs,[0,2,3,1])
		height = out_shape[3]
		width = out_shape[2]
	else:
		height = out_shape[2]
		width = out_shape[1]
	inputs = tf.image.resize_nearest_neighbor(inputs,(new_height,new_width))

	# back to NCHW if needed 
	if data_format =='NCHW':
		inputs = tf.transpose(inputs,[0,3,1,2])

	inputs = tf.identity(inputs,name='upsampled')
	return inputs 



