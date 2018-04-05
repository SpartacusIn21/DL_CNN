#encoding:utf-8
import tensorflow as tf
import numpy as np
from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import sys;
sys.path.append("./toronto_edu_guerzhoy_tf_alexnet/")
from caffe_classes import class_names
#ImageNet数据集尺寸
#读取图片数据
train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]
################################################################################
#Read Image, and change to BGR


im1 = (imread("./toronto_edu_guerzhoy_tf_alexnet/dog.png")[:,:,:3]).astype(float32)
im1 = im1 - mean(im1)
im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]

im2 = (imread("./toronto_edu_guerzhoy_tf_alexnet/dog2.png")[:,:,:3]).astype(float32)
im2[:, :, 0], im2[:, :, 2] = im2[:, :, 2], im2[:, :, 0]

#In Python 3.5, change this to:
#net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
net_data = load("./toronto_edu_guerzhoy_tf_alexnet/bvlc_alexnet.npy").item()

x = tf.placeholder(tf.float32, (None,) + xdim)
################################################################################
#tf.nn.conv2d和tf.layers.conv2d在加载预训练参数方面区别
#For tf.nn.conv2d: 
#filter: A Tensor. Must have the same type as input. A 4-D tensor of shape [filter_height, filter_
#For tf.layers.conv2d:
#filters: Integer, the dimensionality of the output space (i.e. the number of filters in the 

#第一层：卷积层（224x224x3-->55x55x96)
#filters:96x11x11 strides:4 padding="VALID" 
conv1W = tf.Variable(net_data["conv1"][0])
#print("conv1W:%s"%conv1W)
conv1b = tf.Variable(net_data["conv1"][1])
#print("conv1b:%s"%conv1b)
#print("input:%s"%x)
conv1=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,conv1W,[1,4,4,1],padding="SAME"),conv1b))
#print("output/input:%s"%conv1)

#第二层：最大池化层（55x55x96-->27x27x96)
#pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[3,3], strides=2)
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

#第三层：卷积层(27x27x96->27x27x256)
#filters:256x96x96 strides:1 padding="SAME" 
#预训练参数conv2W 维度是针对原论文中两个GPU生成的,维度是256个5x5x48，需要重复一遍数据生成维度256个5x5x96核函数,conv2b维度是5x5x96
conv2W = tf.Variable(net_data["conv2"][0])
conv2W = tf.concat([conv2W ,conv2W],2)
#print("conv2W:%s"%conv2W)
conv2b = tf.Variable(net_data["conv2"][1])
#print("conv2b:%s"%conv2b)
conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool1,conv2W,[1,1,1,1],padding="SAME"),conv2b))
#print("output/input:%s"%conv2)

#第四层：最大池化层(27x27x256-->13x13x256)
#pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)
pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

#第五层：卷积层(13x13x256->13x13x384)
#filters:384x3x3 strides:1 padding="SAME" 
conv3W = tf.Variable(net_data["conv3"][0])
#conv3W这里group=1
#conv3W = tf.concat([conv3W ,conv3W],2)
conv3b = tf.Variable(net_data["conv3"][1])
conv3=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool2,conv3W,[1,1,1,1],padding="SAME"),conv3b))

#第六层：卷积层(13x13x384->13x13x384)
#filters:384x3x3 strides:1 padding="SAME" 
conv4W = tf.Variable(net_data["conv4"][0])
conv4W = tf.concat([conv4W ,conv4W],2)
conv4b = tf.Variable(net_data["conv4"][1])
conv4=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3,conv4W,[1,1,1,1],padding="SAME"),conv4b))

#第七层：卷积层(13x13x384->13x13x256)
#filters:256x3x3 strides:1 padding="SAME" 
conv5W = tf.Variable(net_data["conv5"][0])
conv5W = tf.concat([conv5W ,conv5W],2)
conv5b = tf.Variable(net_data["conv5"][1])
conv5=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4,conv5W,[1,1,1,1],padding="SAME"),conv5b))

#第八层：最大池化层(13x13x256-->6x6x256)
#pool3=tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)
pool3 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

#全连接层，只有全连接层才会进行L1和L2正则化
#fc6
#fc(4096, name='fc6')
fc6W = tf.Variable(net_data["fc6"][0])
fc6b = tf.Variable(net_data["fc6"][1])
#将pool3拍成pool3所有维度相乘后的长度
fc6 = tf.nn.relu_layer(tf.reshape(pool3, [-1, int(prod(pool3.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
fc7W = tf.Variable(net_data["fc7"][0])
fc7b = tf.Variable(net_data["fc7"][1])
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
#fc(1000, relu=False, name='fc8')
fc8W = tf.Variable(net_data["fc8"][0])
fc8b = tf.Variable(net_data["fc8"][1])
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

#---------------------------网络结束---------------------------
prob = tf.nn.softmax(fc8)
#开始训练
sess=tf.InteractiveSession()  
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

t = time.time()
output = sess.run(prob, feed_dict = {x:[im1,im2]})
################################################################################

#Output:


for input_im_ind in range(output.shape[0]):
    inds = argsort(output)[input_im_ind,:]
    print "Image", input_im_ind
    for i in range(5):
        print class_names[inds[-1-i]], output[input_im_ind, inds[-1-i]]

print time.time()-t
