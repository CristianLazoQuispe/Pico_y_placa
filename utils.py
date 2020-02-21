import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from PIL import Image, ImageDraw
import cv2
import time
#import dlib
import operator
from sort import Sort
import numpy as np
import threading
import random
import warnings

import json
import datetime
import requests
import json
import cv2
import base64

"""ssd.py

This module implements the TrtSSD class.
"""


import ctypes

import numpy as np
import cv2
import tensorflow as tf
import tensorrt as trt
import pycuda.driver as cuda

lista_clases = ['bicycle',
    'car',
    'motorcycle',
    'bus',
    'train',
    'truck']


def _preprocess_trt(img, shape=(300, 300)):
    """Preprocess an image before TRT SSD inferencing."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, shape)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img = (2.0/255.0) * img - 1.0
    return img


def _postprocess_trt(img, output, conf_th, output_layout):
    """Postprocess TRT SSD output."""
    img_h, img_w, _ = img.shape
    boxes, confs, clss = [], [], []
    for prefix in range(0, len(output), output_layout):
        #index = int(output[prefix+0])
        conf = float(output[prefix+2])
        if conf < conf_th:
            continue
        x1 = int(output[prefix+3] * img_w)
        y1 = int(output[prefix+4] * img_h)
        x2 = int(output[prefix+5] * img_w)
        y2 = int(output[prefix+6] * img_h)
        cls = int(output[prefix+1])
        boxes.append((x1, y1, x2, y2))
        confs.append(conf)
        clss.append(cls)
    return boxes, confs, clss

lista_clases = ['bicycle',
    'car',
    'motorcycle',
    'bus',
    'train',
    'truck']

def get_name_license(alpr,img_rgb,tresh = 70):
    bandera = False
    name_license = ''
    confidence = 0
    try:
        results = alpr.recognize_ndarray(img_rgb)
        if (len( results['results'])>0):
            #if(len(results['results'][0]['candidates'])>0 and results['results'][0]['confidence']>=tresh):
            if(results['results'][0]['confidence']>=tresh):
                bandera = True
            name_license = results['results'][0]['plate']
            confidence = results['results'][0]['confidence']
        return bandera,name_license,confidence
    except:
        return bandera,name_license,confidence


class TrtSSD(object):
    """TrtSSD class encapsulates things needed to run TRT SSD."""

    def _load_plugins(self):
        if trt.__version__[0] < '7':
            ctypes.CDLL("ssd/libflattenconcat.so")
        trt.init_libnvinfer_plugins(self.trt_logger, '')

    def _load_engine(self):
        TRTbin = 'ssd/TRT_%s.bin' % self.model
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _create_context(self):
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        return self.engine.create_execution_context()

    def __init__(self, model, input_shape, output_layout=7):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model = model
        self.input_shape = input_shape
        self.output_layout = output_layout
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self._load_plugins()
        self.engine = self._load_engine()

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.context = self._create_context()

    def __del__(self):
        """Free CUDA memories."""
        try:
            del self.stream
            del self.cuda_outputs
            del self.cuda_inputs
        except:
            a=1+1

    def detect(self, img, conf_th=0.3):
        """Detect objects in the input image."""
        img_resized = _preprocess_trt(img, self.input_shape)
        np.copyto(self.host_inputs[0], img_resized.ravel())

        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(
            self.host_outputs[1], self.cuda_outputs[1], self.stream)
        cuda.memcpy_dtoh_async(
            self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        output = self.host_outputs[0]
        return _postprocess_trt(img, output, conf_th, self.output_layout)



def _preprocess_tf(img, shape=(300, 300)):
    """Preprocess an image before TensorFlow SSD inferencing."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, shape)
    return img

import itertools
# import the necessary packages
import numpy as np

def non_max_suppression_fast(boxes, overlapThresh):
   # if there are no boxes, return an empty list
   if len(boxes) == 0:
      return []

   # if the bounding boxes integers, convert them to floats --
   # this is important since we'll be doing a bunch of divisions
   if boxes.dtype.kind == "i":
      boxes = boxes.astype("float")
#  
   # initialize the list of picked indexes   
   pick = []

   # grab the coordinates of the bounding boxes
   x1 = boxes[:,0]
   y1 = boxes[:,1]
   x2 = boxes[:,2]
   y2 = boxes[:,3]

   # compute the area of the bounding boxes and sort the bounding
   # boxes by the bottom-right y-coordinate of the bounding box
   area = (x2 - x1 + 1) * (y2 - y1 + 1)
   idxs = np.argsort(y2)

   # keep looping while some indexes still remain in the indexes
   # list
   while len(idxs) > 0:
      # grab the last index in the indexes list and add the
      # index value to the list of picked indexes
      last = len(idxs) - 1
      i = idxs[last]
      pick.append(i)

      # find the largest (x, y) coordinates for the start of
      # the bounding box and the smallest (x, y) coordinates
      # for the end of the bounding box
      xx1 = np.maximum(x1[i], x1[idxs[:last]])
      yy1 = np.maximum(y1[i], y1[idxs[:last]])
      xx2 = np.minimum(x2[i], x2[idxs[:last]])
      yy2 = np.minimum(y2[i], y2[idxs[:last]])

      # compute the width and height of the bounding box
      w = np.maximum(0, xx2 - xx1 + 1)
      h = np.maximum(0, yy2 - yy1 + 1)

      # compute the ratio of overlap
      overlap = (w * h) / area[idxs[:last]]

      # delete all indexes from the index list that have
      idxs = np.delete(idxs, np.concatenate(([last],
         np.where(overlap > overlapThresh)[0])))

   # return only the bounding boxes that were picked using the
   # integer data type
   return boxes[pick].astype("int")
def _postprocess_tf(img, boxes, scores, classes, conf_th):
    """Postprocess TensorFlow SSD output."""
    h, w, _ = img.shape
    out_boxes = boxes[0] * np.array([h, w, h, w])
    out_boxes = out_boxes.astype(np.int32)
    out_boxes = out_boxes[:, [1, 0, 3, 2]]  # swap x's and y's
    out_confs = scores[0]
    out_clss = classes[0].astype(np.int32)

    # only return bboxes with confidence score above threshold
    mask = np.where(out_confs >= conf_th)
    return out_boxes[mask], out_confs[mask], out_clss[mask]


class TfSSD(object):
    """TfSSD class encapsulates things needed to run TensorFlow SSD."""

    def __init__(self, model, input_shape):
        self.model = model
        self.input_shape = input_shape

        # load detection graph
        ssd_graph = tf.Graph()
        with ssd_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile('ssd/%s.pb' % model, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')

        # define input/output tensors
        self.image_tensor = ssd_graph.get_tensor_by_name('image_tensor:0')
        self.det_boxes = ssd_graph.get_tensor_by_name('detection_boxes:0')
        self.det_scores = ssd_graph.get_tensor_by_name('detection_scores:0')
        self.det_classes = ssd_graph.get_tensor_by_name('detection_classes:0')

        # create the session for inferencing
        self.sess = tf.Session(graph=ssd_graph)

    def __del__(self):
        self.sess.close()

    def detect(self, img, conf_th):
        img_resized = _preprocess_tf(img, self.input_shape)
        boxes, scores, classes = self.sess.run(
            [self.det_boxes, self.det_scores, self.det_classes],
            feed_dict={self.image_tensor: np.expand_dims(img_resized, 0)})
        return _postprocess_tf(img, boxes, scores, classes, conf_th)

COCO_CLASSES_LIST = [
    'background',  # was 'unlabeled'
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'street sign',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'hat',
    'backpack',
    'umbrella',
    'shoe',
    'eye glasses',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'plate',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'mirror',
    'dining table',
    'window',
    'desk',
    'toilet',
    'door',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'blender',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]

EGOHANDS_CLASSES_LIST = [
    'background',
    'hand',
]


def get_cls_dict(model):
    """Get the class ID to name translation dictionary."""
    if model == 'coco':
        cls_list = COCO_CLASSES_LIST
    elif model == 'egohands':
        cls_list = EGOHANDS_CLASSES_LIST
    else:
        raise ValueError('Bad model name')
    return {i: n for i, n in enumerate(cls_list)}

def get_time_now():
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_time = datetime.time(int(current_time[:2]),int(current_time[3:5]),int(current_time[6:8]))
    return current_time


def correction_ilu_01(BGR):
    B,G,R = cv2.split(BGR)
    #Create a CLAHE object: The image is divided into small block 8x8 which they are equalized as usual.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    #Applying this method to each channel of the color image
    output_2R = clahe.apply(R)
    output_2G = clahe.apply(G)
    output_2B = clahe.apply(B)
    #mergin each channel back to one
    img_output = cv2.merge((output_2B,output_2G,output_2R))
    return img_output

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )



