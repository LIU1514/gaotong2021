import cv2
import numpy as np
import os
import random
'''
VOC_path = '/home/data/7/car'
VOC_raw_path = '/usr/local/ev_sdk/raw'
raw_list_file = '/usr/local/ev_sdk/raw_list.txt'

'''
VOC_path = '/home/data/7/car'
VOC_raw_path = '/usr/local/ev_sdk/raw'
 
raw_list_file = '/usr/local/ev_sdk/raw_list.txt'
raw_file_names = ''

if not os.path.exists(VOC_raw_path):
    os.makedirs(VOC_raw_path)
 

imgs = os.listdir(VOC_path)
for i, file_name in enumerate(imgs):
    img = cv2.imread(os.path.join(VOC_path, file_name))
    resized_img = cv2.resize(img, (100, 100), cv2.INTER_LINEAR)
 


    mean,std = cv2.meanStdDev(img)
    mean = mean.reshape(1,3)
    std = std.reshape(1,3)

    resized_img = (resized_img-mean)/(0.000001 + std)
    resized_img_data = np.array(resized_img, np.float32)
    raw_file_name = os.path.join(VOC_raw_path, str(i) + '.raw')
    resized_img_data.tofile(raw_file_name)

    raw_file_names += (raw_file_name + '\n')
    if i == 5:
        break


with open(raw_list_file, 'w') as f:
    f.write(raw_file_names)
f.close()