'''
                       _oo0oo_
                      o8888888o
                      88" . "88
                      (| -_- |)
                      0\  =  /0
                    ___/`---'\___
                  .' \\|     |// '.
                 / \\|||  :  |||// \
                / _||||| -:- |||||- \
               |   | \\\  - /// |   |
               | \_|  ''\---/''  |_/ |
               \  .-\__  '-'  ___/-. /
             ___'. .'  /--.--\  `. .'___
          ."" '<  `.___\_<|>_/___.' >' "".
         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
         \  \ `_.   \_ __\ /__ _/   .-` /  /
     =====`-.____`.___ \_____/___.-`___.-'=====
                       `=---='


     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

           佛祖保佑       永不宕机     永无BUG

       佛曰:  
               写字楼里写字间，写字间里程序员；  
               程序人员写程序，又拿程序换酒钱。  
               酒醒只在网上坐，酒醉还来网下眠；  
               酒醉酒醒日复日，网上网下年复年。  
               但愿老死电脑间，不愿鞠躬老板前；  
               奔驰宝马贵者趣，公交自行程序员。  
               别人笑我忒疯癫，我笑自己命太贱；  
               不见满街漂亮妹，哪个归得程序员？
'''



from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf

 
#import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import pandas as pd
import argparse
import random
import pickle
import cv2
import os
import cnn
import dataset
 
 
 
# 将所有的图片resize成100*100


w = 100
h = 100
c = 3
 
IMAGE_DIMS = (100, 100, 3)

# initialize the data and labels
data = []
labels = []

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
#imagePaths = sorted(list(paths.list_images(args["dataset"])))
imagePaths = sorted(list(paths.list_images( "/home/data/7/")))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)
 
	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
	data.nbytes / (1024 * 1000.0)))

# binarize the labels
 
lb = LabelBinarizer()
labels = lb.fit_transform(labels)  
image_list=data 
onehotlabels=labels    
label_list = np.argmax(onehotlabels, axis=1)  

 
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
#x = tf.reshape(x, [-1, 100, 100, 3],name='input') 
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

logits,pred = cnn.simple_cnn(x)
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


 

# 训练和测试数据，可将n_epoch设置更大一些
n_epoch = 11
batch_size = 16
def train():
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch in range(n_epoch):
        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in dataset.minibatches(data, label_list, batch_size, shuffle=True):
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err
            train_acc += ac
            n_batch += 1

        '''
        print('Epoch %d - train loss: %f'%(epoch, (train_loss / n_batch)))
        print('Epoch %d - train acc: %f'%(epoch,train_acc / n_batch))
        '''

        print('Epoch:', epoch, '| train loss: %.4f' % (train_loss / n_batch), '| test accuracy: %.2f' % (train_acc / n_batch))

    # validation
    print("run over......")
    # #save ckpt
    export_dir = '/project/train/models/final'
    #saver = tf.train.Saver()
    step = 200
    # if os.path.exists(export_dir):
    #     os.system("rm -rf " + export_dir)
    if not os.path.isdir(export_dir): 
        os.makedirs(export_dir)

    checkpoint_file = os.path.join(export_dir, 'model.ckpt')
    saver.save(sess, checkpoint_file, global_step=step)
    #saver.save(sess, checkpoint_file)
    
    def ckptToPb():
        checkpoint_file = os.path.join(export_dir, 'model.ckpt-200.meta')
        ckpt = tf.train.get_checkpoint_state(export_dir)
        print("model ", ckpt.model_checkpoint_path)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
        graph = tf.get_default_graph()
        with tf.Session() as sess:
            saver.restore(sess,ckpt.model_checkpoint_path)
            
            input_image = tf.get_default_graph().get_tensor_by_name("input:0")
            fc0_output = tf.get_default_graph().get_tensor_by_name("y_conv:0")
            sess.run(tf.global_variables_initializer())
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), ['y_conv'])
            model_name = os.path.join(export_dir, 'model200.pb')
            with tf.gfile.GFile(model_name, "wb") as f:  
                f.write(output_graph_def.SerializeToString()) 
    print('model saved')
    ckptToPb() 



train()

print("[INFO] serializing label binarizer...")
f = open('/project/train/models/final/lb.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()
