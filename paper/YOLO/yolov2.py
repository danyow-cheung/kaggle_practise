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
        if x3<x1:
            if x4<x1:
                return 0 
            else:
                return min(x2,x4) - x1 
        
        else:
            if x2<x3:
                return 0 
            else:
                return min(x2,x4) - x3 
    
    def bbox_iou(self,box1,box2):
        intersect_w = self._interval_overlap([box1.xmin,box1.xmax],[box2.xmin,box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin,box1.ymax],[box2.ymin,box2.ymax])
        intersect = intersect_w*intersect_h

        w1,h1 = box1.xmax - box1.xmin,box1.ymax-box1.ymin
        w2,h2 = box2.xmax - box2.xmin,box2.ymax - box2.ymin

        union = w1*h1 + w2*h2 - intersect
        return float(intersect)/union
    
    def find(self,center_w,center_h):
        # find the anchor that best predicts this box 
        best_anchor = -1 
        max_iou = -1
        # each anchor box is specialized to have a certain shape 
        # eg:flat large rectangle or small square 
        shifted_box = BoundBox(0,0,center_w,center_h)
        # for given object,find the best anchor box 
        for i in range(len(self.anchors)):
            anchor = self.anchors[i]
            iou = self.bbox_iou(shifted_box,anchor)
            if max_iou<iou:
                best_anchor= i 
                max_iou = iou
        return (best_anchor,max_iou)
    

class BoundBox:
    def __init__(self,xmin,ymin,xmax,ymax,confidence=None,classes=None) :
        self.xmin = xmin
        self.ymin = ymin 
        self.xmax = xmax
        self.ymax = ymax
        # the code below are used during inference
        # probability
        self.confidence = confidence

        # class probabilities 
        self.set_class(classes)

    def set_class(self,classes):
        self.classes = classes
        self.label = np.argmax(self.classes)

    def get_label(self):
        return self.label
    
    def get_score(self):
        return(self.classes[self.label])

'''sample usage of the BESTANCHORBOXFINDER class '''
# Anchor box width and height found in https://fairyonice.github.io/Part_1_Object_Detection_with_Yolo_for_VOC_2014_data_anchor_box_clustering.html
_ANCHORS01 = np.array([0.08285376, 0.13705531,
                       0.20850361, 0.39420716,
                       0.80552421, 0.77665105,
                       0.42194719, 0.62385487])
print(".."*40)
print("The three example anchor boxes:")
count = 0
for i in range(0,len(_ANCHORS01),2):
    print("anchor box index={}, w={}, h={}".format(count,_ANCHORS01[i],_ANCHORS01[i+1]))
    count += 1
print(".."*40)   
print("Allocate bounding box of various width and height into the three anchor boxes:")  
babf = BestAnchorBoxFinder(_ANCHORS01)
for w in range(1,9,2):
    w /= 10.
    for h in range(1,9,2):
        h /= 10.
        best_anchor,max_iou = babf.find(w,h)
        print("bounding box (w = {}, h = {}) --> best anchor box index = {}, iou = {:03.2f}".format(
            w,h,best_anchor,max_iou))

def rescale_centerxy(obj,config):
    '''
    Arguments:
        obj:        dictionary containing xmin,xmax,ymin,ymax
        config:     dictionary containing IMAGE_W,GRID_W,IMAGE_H and GRID_H
    
    '''
    center_x = 0.5*(obj['xmin'] + obj['xmax'])
    center_x = center_x/(float(config["IMAGE_W"]) / config['GRID_W'])
    center_y = 0.5*(obj['ymin'] + obj['ymax'])
    center_y = center_y/(float(config['IMAGE_H']) / config['GRID_H'])
    return (center_x,center_y)

def rescale_cebterwh(obj,config):
    '''
    Arguments:
        obj:        dictionary containing xmin,xmax,ymin,ymax
        config:     dictionary containing IMAGE_W,GRID_W,IMAGE_H and GRID_H
    '''
    # unit :grid cell
    center_w = (obj['xmax'] - obj['xmin'])/(float(config['IMAGE_W'])/config['GRID_W'])
    # unit :grid cell
    center_h = (obj['ymax'] - obj['ymin'])/(float(config['IMAGE_H'])/config['GRID_H'])
    return(center_w,center_h)

'''sample usage'''
obj    = {'xmin': 150, 'ymin': 84, 'xmax': 300, 'ymax': 294}
config = {"IMAGE_W":416,"IMAGE_H":416,"GRID_W":13,"GRID_H":13}
center_x, center_y = rescale_centerxy(obj,config)
center_w, center_h = rescale_cebterwh(obj,config)

print("cebter_x abd cebter_w should range between 0 and {}".format(config["GRID_W"]))
print("cebter_y abd cebter_h should range between 0 and {}".format(config["GRID_H"]))

print("center_x = {:06.3f} range between 0 and {}".format(center_x, config["GRID_W"]))
print("center_y = {:06.3f} range between 0 and {}".format(center_y, config["GRID_H"]))
print("center_w = {:06.3f} range between 0 and {}".format(center_w, config["GRID_W"]))
print("center_h = {:06.3f} range between 0 and {}".format(center_h, config["GRID_H"]))


'''Define a custon Batch generator to get a batch of 16 images and its corresponding bounding boxes'''
# from tensorflow.keras.utils import Sequence
from keras.utils import Sequence

class SimpleBatchGenerator(Sequence):
    def __init__(self,images,config,norm=None,shuffle=True):
        '''
        config : dictionary containing necessary hyper parameters for traning. e.g., 
            {
            'IMAGE_H'         : 416, 
            'IMAGE_W'         : 416,
            'GRID_H'          : 13,  
            'GRID_W'          : 13,
            'LABELS'          : ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle', 
                                  'bus',        'car',      'cat',  'chair',     'cow',
                                  'diningtable','dog',    'horse',  'motorbike', 'person',
                                  'pottedplant','sheep',  'sofa',   'train',   'tvmonitor'],
            'ANCHORS'         : array([ 1.07709888,   1.78171903,  
                                        2.71054693,   5.12469308, 
                                        10.47181473, 10.09646365,  
                                        5.48531347,   8.11011331]),
            'BATCH_SIZE'      : 16,
            'TRUE_BOX_BUFFER' : 50,
            }
        
        '''

        self.config = config
        self.config['BOX'] = int(len(self.config['ANCHORS'])/2)
        self.config['CLASS']  = len(self.config['LABELS'])
        self.images = images
        self.bestAnchorBoxFinder = BestAnchorBoxFinder(config['ANCHORS'])
        self.imageReader = ImageReader(config['IMAGE_H'],config['IMAGE_W'],norm=norm)
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.images)
        
        def __len__(self):
            return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))
        
        def __getitem___(self,idx):
            '''
            Arguments:
                idx: non-negeative integer value 
            Returns:
                x_batch: the numpy array of shape (BATCH_SIZE,IMAGE_H,IMAGE_W,N,channels)
                        x_batch[iframe,:,:,:] contains a iframeth frame of size  (IMAGE_H,IMAGE_W).

                y_batch: the numpy array of shape (BATCH_SIZE,GRID_H,GRID_W,BOX,4+1+Nclassess)
                        Box = the number of anchor boxes

                        y_batch[iframe,igird_h,igrid_w,ianchor,:4] contains a (center_x,center_y,center_w,center_h)
                        of ianchorth anchor at grid cell = (igrid_h,igrid_w) if the object exists in this (grid cell,anchor)
                        pair,else they simply contain 0 

                        y_batch[iframe,igrid_h,igrid_w,ianchor,5 + iclass] contains 1 if the iclass^th 
                        class object exists in this (grid cell, anchor) pair, else it contains 0.

                b_batch: the numpy array of shape(BATCH_SIZE,1,1,1,TRUE_BOX_BUFFER,4)
                        b_batch[iframe,1,1,1,ibuffer,ianchor,:] contains ibufferth object's (center_X,center_y,center_w,center_h) in iframe frame
                        If ibuffer > N objects in iframeth frame ,then the values are simply 0 
                        TRUE_BOX_BUFFER has to be some large number, so that the frame with the 
                        biggest number of objects can also record all objects.

                        The order of the objects do not matter.

                        This is just a hack to easily calculate loss. 
            '''
            l_bound = idx*self.config['BATCH_SIZE']

            r_bound =  (idx+1)*self.config['BATCH_SIZE']

            if r_bound>len(self.images):
                r_bound = len(self.images)
                l_bound = r_bound - self.config['BATCH_SIZE']
            instance_count = 0 

            