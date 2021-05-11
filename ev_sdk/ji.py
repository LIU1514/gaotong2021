'''
Author: your name
Date: 2021-05-11 10:49:31
LastEditTime: 2021-05-11 10:50:20
LastEditors: your name
Description: In User Settings Edit
FilePath: \raw\ev_sdk\ji.py
'''

from __future__ import print_function
import tensorflow as tf
import logging as log
import json
import cv2
import os
import pickle
#import sklearn
import numpy as np

log.basicConfig(level=log.DEBUG)

sess = None
input_w, input_h, input_c, input_n = (100, 100, 3, 1)

label_id_map = pickle.loads(open('/usr/local/ev_sdk/lb.pickle', "rb").read())
    
def init():
    save_model =  "/usr/local/ev_sdk/model/model200.pb"
    if not os.path.isfile(save_model):
        log.error(f"{save_model} does not exist")

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(save_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')    
 
    log.info('Initializing session...')
    global sess
    sess = tf.Session(graph=detection_graph)
    return detection_graph
def process_image(net, input_image, args=None):
    """Do inference to analysis input_image and get output

    Attributes:
        net: model handle
        input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
        args: optional args

    Returns: process result

    """

    # ------------------------------- Prepare input -------------------------------------
    if not net or input_image is None:
        log.error('Invalid input args')
        return None
    ih, iw, _ = input_image.shape

    if ih != input_h or iw != input_w:
        input_image = cv2.resize(input_image, (input_w, input_h))
    input_image = np.expand_dims(input_image, axis=0)

    # --------------------------- Performing inference ----------------------------------
    # Extract image tensor
    image_tensor = net.get_tensor_by_name('input:0')
    # Extract detection boxes, scores, classes, number of detections

    out_softmax = net.get_tensor_by_name("y_conv:0")

    # Actual detection.
    img_out_softmax= sess.run(out_softmax,feed_dict={image_tensor: input_image})

    detect_objs = []
    prediction_labels = np.argmax(img_out_softmax)
    data = {'class':label_id_map.classes_[prediction_labels]}
    print(label_id_map.classes_[prediction_labels])
    print(data)
    return json.dumps(data, indent=4)