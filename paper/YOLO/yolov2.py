import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 

'''Data Preprocessing
> !wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
> !tar xvf VOCtrainval_11-May-2012.tar
> rm VOCtrainval_11-May-2012.tar
'''

'''Define a function that parses the annotations from the XML files and stores it in an array
'''
import os 
import xml.etree.cElementTree as ET
# 通过路径名构建数据集的树
def parse_annotation(ann_dir,img_dir,labels=[]):
    all_imgs = []
    seen_labels = {}
    for ann in sorted(os.listdir(ann_dir)):
        if "xml" not in ann:
            continue
        img ={'object':[]}
        tree = ET.parse(ann_dir+ann)
        for elem in tree.iter():
            if "filename" in elem.tag:
                path_to_image = img_dir+elem.text
                img['filename'] = path_to_image
                if not os.path.exists(path_to_image):
                    assert ValueError,"file doesn't exist\n{}".format(path_to_image)
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text
                        if len(labels)>0 and obj['name'] not in labels:
                            break
                        else:
                            img['object']+=[obj]
                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] =  1
                        
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))
                    
            if len(img['object'])>0:
                all_imgs+=[img]
    return all_imgs,seen_labels


'''Anchor box blog:https://fairyonice.github.io/Part_1_Object_Detection_with_Yolo_for_VOC_2014_data_anchor_box_clustering.html'''
# parse annotations
train_image_folder = "VOCdevkit/VOC2012/JPEGImages/"
train_annot_folder = "VOCdevkit/VOC2012/Annotations/"

ANCHORS = np.array([1.07709888,  1.78171903,  # anchor box 1, width , height
                    2.71054693,  5.12469308,  # anchor box 2, width,  height
                   10.47181473, 10.09646365,  # anchor box 3, width,  height
                    5.48531347,  8.11011331]) # anchor box 4, width,  height
LABELS = ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle', 
          'bus',        'car',      'cat',  'chair',     'cow',
          'diningtable','dog',    'horse',  'motorbike', 'person',
          'pottedplant','sheep',  'sofa',   'train',   'tvmonitor']

train_image, seen_train_labels = parse_annotation(train_annot_folder,train_image_folder, labels=LABELS)
print("N train = {}".format(len(train_image)))
'''Define a ImageReader class to process an image,it takes in an image and returns 
the resized image and all the objects in the image'''
import copy 
import cv2 

class ImageReader(object):
    def __init__(self,IMAGE_H,IMAGE_W,norm=None):
        self.IMAGE_H = IMAGE_H
        self.IMAGE_W = IMAGE_W
        self.norm = norm
    
    def encode_core(self,image,reorder_rgb = True):
        image = cv2.resize(image,(self.IMAGE_H,self.IMAGE_W))
        if reorder_rgb:
            image = image[:,:,::-1]
        if self.norm is not None:
            image = self.norm(image)
        return(image)
    
    def fit(self,train_instance):
        '''read in and resize the image,annotations are resized accordingly
        Arguments:
            train_instance: dictionary containing filename,height,width and object'''
        if not isinstance(train_instance,dict):
            train_instance = {"filename":train_instance}
        
        image_name = train_instance['filename']
        image = cv2.imread(image_name)
        h,w,c = image.shape 
        if image is None:
            print("Can't find ",image_name)
        image = self.encode_core(image,reorder_rgb=True)
        if 'object' in train_instance.keys():
            all_objs = copy.deepcopy(train_instance['object'])
            # fix object's postition and size 
            for obj in all_objs:
                for attr in ['xmin','xmax']:
                    obj[attr] = int(obj[attr]*float(self.IMAGE_W)/w)
                    obj[attr] = max(min(obj[attr],self.IMAGE_W),0)
                
                for attr in ['ymin','ymax']:
                    obj[attr] = int(obj[attr] * float(self.IMAGE_H)/h)
                    obj[attr] = max(min(obj[attr],self.IMAGE_H),0)
        else:
            return image
        return image,all_objs

# sample usage of the ImageReader class 
def normalize(image):
    return image/255.

print("Input")
timage = train_image[0]
for key, v in timage.items():
    print("  {}: {}".format(key,v))
print("*"*30)
print("Output")
inputEncoder = ImageReader(IMAGE_H=416,IMAGE_W=416, norm=normalize)
image, all_objs = inputEncoder.fit(timage)
print("          {}".format(all_objs))
plt.imshow(image)
plt.title("image.shape={}".format(image.shape))
plt.show()


'''Define BestAnchorBoxFinder which finds the best anchor box for a particular object
This is done by finding the anchor box with the highest IOU(Intersection over Union) with the bounding box of the object.
'''
class BestAnchorBoxFinder(object):
    '''Anchors: a np.array of even number length '''
    def __init__(self,ANCHORS):
        self.anchors = [BoundBox(0,0,ANCHORS[2*i],ANCHORS[2*i+1]) for i in range(int(len(ANCHORS)//2))]

    def _interval_overlap(self,interval_a,interval_b):
        x1,x2 = interval_a
        x3,x4 = interval_b
        