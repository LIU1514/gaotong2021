import cv2
import numpy as np
import os
import random

#VOC_path = '/home/data/7/car'
VOC_raw_path = '/usr/local/ev_sdk/raw'
 
raw_list_file = '/usr/local/ev_sdk/raw_list.txt'
raw_file_names = ''

if not os.path.exists(VOC_raw_path):
    os.makedirs(VOC_raw_path)
 

imgs = os.listdir(VOC_raw_path)
for i, file_name in enumerate(imgs):
    raw_file_name = os.path.join(VOC_raw_path, file_name)
 
    raw_file_names += (raw_file_name + '\n')
 
with open(raw_list_file, 'w') as f:
    f.write(raw_file_names)
f.close()