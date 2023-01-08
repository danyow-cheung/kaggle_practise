import tensorflow as tf 
import matplotlib.pyplot as plt 
import cv2 
import numpy as np
from tensorflow import keras
import keras.backend as K

'''
接下來，我們處理註釋並將標籤寫入文本文件。 與 XML 相比，文本文件更易於使用。
'''
import argparse
import xml.etree.ElementTree as ET 
import os 

parser = argparse.ArgumentParser(description='Build Annotations')
parser.add_argument('dir',default="..",help='Annotations.')

sets = [('2007','train'),('2007','val'),('2007','test')]

class_num =  {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
               'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
               'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
               'sofa': 17, 'train': 18, 'tvmonitor': 19}

def convert_annotation(year,image_id,f):
	in_file = os.path.join('VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id))
	tree = ET.parse(in_file)
	root = tree.getroot()

	for obj in root.iter('object'):
		difficult = obj.find('diffcult').text 
		cls = obj.find('name').text 
		classes = list(class_num.keys())
		if cls not in classes or int(difficult)==1:
			continue 

		cls_id = classes.index(cls)
		xmlbox = obj.find('bndbox')
		b = (int(xmlbox.find('xmin').text),int(xmlbox.find('ymin').text),
			int(xmlbox.find('xmax').text),int(xmlbox.find('ymax').text
				))
		f.write(" "+",".join([str(a) for a in b])+","+str(cls_id))

for year ,image_set in sets:
	print(year,image_set)
	with open(os.path.join('VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set)), 'r') as f:
		image_ids = f.read().strip().split()

	with open(os.path.join("VOCdevkit", '%s_%s.txt' % (year, image_set)), 'w') as f2:

		for image_id in image_ids:

			f2.write('%s/VOC%s/JPEGImages/%s.jpg' % ("VOCdevkit", year, image_id))
			convert_annotation(year,image_id,f2)
			f2.write('\n')

'''接下來，我要添加一個函數來準備輸入和輸出。 
輸入是 (448, 448, 3) 圖像，
輸出是 (7, 7, 30) 張量。 
輸出基於 S x S x (B * 5 +C)。'''
def read(image_path,label):
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image,cv2.COLOR_BAYER_BGR2RGB)
	image_h,image_w = image.shape[0:2]
	image = cv2.resize(image,(448,448))
	image = image/255 

	label_matrix = np.zeros([7,7,30])
	for l in label:
		l = l.split(',')
		l = np.array(l,dtype=np.int)
		xmin = l[0]
		ymin = l[1]
		xmax = l[2]
		ymax = l[3]
		cls = l[4]

		x = (xmin+xmax)/2/image_w
		y =  (ymin+ymax)/2/image_h
		w = (xmax-xmin)/image_w
		h = (ymax-ymin)/image_h

		loc = [7*x,7*y]
		loc_i = int(loc[1])
		loc_j = int(loc[0])

		y = loc[1] - loc_i 
		x = loc[0] - loc_j

		if label_matrix[loc_i,loc_j,24]==0:
			label_matrix[loc_i,loc_j,cls]=1
			label_matrix[loc_i,loc_j,20:24] = [x,y,w,h]
			label_matrix[loc_i,loc_j,24] = 1 # response 
	return image,label_matrix



'''
接下來，定義一個自定義生成器，它返回一批輸入和輸出。
'''
class Generator(keras.utils.Sequence):
	def __init__(self,images,labels,batch_size) :
		super().__init__()
		self.images = images 
		self.labels = labels
		self.batch_size = batch_size

	
	def __len__(self):
		return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)


	def __getitem__(self,idx):
		batch_x = self.images[idx*self.batch_size:(idx+1)*self.batch_size]
		batch_y = self.labels[idx*self.batch_size:(idx+1)*self.batch_size]

		train_image = []
		trian_label = []
		for i in range(0,len(batch_x)):
			img_path = batch_x[i]
			label = batch_y[i]

			image,label_matrix = read(img_path,label)
			train_image.append(image)
			trian_label.append(label_matrix)

		return np.array(train_image),np.array(trian_label)
	
'''
下面的代碼片段準備了帶有輸入和輸出的數組。
'''
train_datasets =[]
val_datasets = []


with open(os.path.join("VOCdevkit", '2007_train.txt'), 'r') as f:
    train_datasets = train_datasets + f.readlines()
with open(os.path.join("VOCdevkit", '2007_val.txt'), 'r') as f:
    val_datasets = val_datasets + f.readlines()

X_train = []
Y_train = []
X_val = []
Y_val = []

for item in train_datasets:
	item = item.replace("\n","").split(" ")
	X_train.append(item[0])
	arr = []
	for i in range(1,len(item)):
		arr.append(item[i])
	Y_train.append(arr)


for item in val_datasets:
	item = item.replace("\n","").split(" ")
	X_val.append(item[0])
	arr = []
	for i in range(1,len(item)):
		arr.append(item[i])
	Y_val.append(arr)


'''接下來，我們為我們的訓練和驗證集創建生成器實例。'''
batch_size = 4 
my_training_batch_generator = Generator(X_train,Y_train,batch_size)
my_validation_batch_generator = Generator(X_val,Y_val,batch_size)

x_train, y_train = my_training_batch_generator.__getitem__(0)
x_val, y_val = my_training_batch_generator.__getitem__(0)
print(x_train.shape)
print(y_train.shape)

print(x_val.shape)
print(y_val.shape)


'''
我們需要重塑模型的輸出，因此我們為它定義了一個自定義的 Keras 層。
'''

class YOLO_reshape(tf.keras.layers.Layer):
	def __init__(self,target_shape) :
		super(YOLO_reshape).__init__()
		self.target_shape = target_shape
	
	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'target_shape':self.target_shape
		})
		return config
	
	def call(self,input):
		# girds 7x7
		S = [self.target_shape[0],self.target_shape[1]]
		# classes 
		C = 20 
		# number of bounding boxes per grid 
		B = 2 
		idx1= S[0]*S[1]*C 
		idx2 = idx1+S[0]*S[1]*B 

		# class probabilities
		class_probs = K.reshape(input[:,:idx1],(K.shape(input)[0],)+tuple([S[0],S[1],C]))

		class_probs = K.softmax(class_probs)

		# confidence
		confs = K.reshape(input[:, idx1:idx2], (K.shape(input)[0],) + tuple([S[0], S[1], B]))
		confs = K.sigmoid(confs)

		# boxes 
		boxes = K.reshape(input[:, idx2:], (K.shape(input)[0],) + tuple([S[0], S[1], B * 4]))
		boxes = K.sigmoid(boxes)
		
		outputs = K.concatenate([class_probs, confs, boxes])
		
		return outputs

