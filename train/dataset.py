from skimage import io, transform
import os
import numpy as np

# 将所有的图片resize成100*100
w = 100
h = 100
c = 3


# 读取图片
def read_img(path):
    imgs = []
    labels = []
    classs = os.listdir(path)

    for idx, folder in enumerate(classs):
        cate = os.path.join(path, folder)
        for im in os.listdir(cate):
            img_path =os.path.join(cate, im)
            # print('reading the images:%s' % (img_path))
            img = io.imread(img_path)
            img = transform.resize(img, (w, h))
            # with open('tests.txt', 'a') as f:
            #     f.write(img_path+'_'+str(idx)+'\n')
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)



def suffer(data, label):
    # 打乱顺序
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]

    # 将所有数据分为训练集和验证集
    ratio = 0.8
    s = np.int(num_example * ratio)
    x_train = data[:s]
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]
    return x_train,y_train,x_val,y_val

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]