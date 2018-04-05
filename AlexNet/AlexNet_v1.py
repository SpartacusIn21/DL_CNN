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
w=224
h=224
c=3
#读取图片数据
train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]
#-----------------构建网络----------------------
#占位符
x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
		#<tensorflow 1.0 version
        #input_groups = tf.split(3, group, input)
        #kernel_groups = tf.split(3, group, kernel)
		#>= tensorflow 1.0 version
        input_groups = tf.split(input,group,3)
        kernel_groups = tf.split(kernel, group,3)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
		#<tensorflow 1.0 version
        #conv = tf.concat(3, output_groups)
		#>=tensorflow 1.0 version
        conv = tf.concat(output_groups,3)
    return  tf.nn.bias_add(conv, biases)
################################################################################
#Read Image, and change to BGR


im1 = (imread("./toronto_edu_guerzhoy_tf_alexnet/dog.png")[:,:,:3]).astype(float32)
#im1 = (imread("./toronto_edu_guerzhoy_tf_alexnet/laska.png")[:,:,:3]).astype(int32)
im1 = im1 - mean(im1)
im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]

im2 = (imread("./toronto_edu_guerzhoy_tf_alexnet/dog2.png")[:,:,:3]).astype(float32)
#im2 = (imread("poodle.png")[:,:,:3]).astype(int32)
im2[:, :, 0], im2[:, :, 2] = im2[:, :, 2], im2[:, :, 0]

#In Python 3.5, change this to:
#net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
net_data = load("./toronto_edu_guerzhoy_tf_alexnet/bvlc_alexnet.npy").item()

x = tf.placeholder(tf.float32, (None,) + xdim)
################################################################################
#For tf.nn.conv2d: 
#filter: A Tensor. Must have the same type as input. A 4-D tensor of shape [filter_height, fil
#For tf.layers.conv2d:
#filters: Integer, the dimensionality of the output space (i.e. the number of filters in the 

#第一层：卷积层（224x224x3-->55x55x96)
#filters:96x11x11 strides:4 padding="VALID" 
conv1W = tf.Variable(net_data["conv1"][0])
print("conv1W:%s"%conv1W)
conv1b = tf.Variable(net_data["conv1"][1])
print("conv1b:%s"%conv1b)
print(x)
conv1=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,conv1W,[1,4,4,1],padding="VALID"),conv1b))
print("conv1:%s"%conv1)
conv1 = tf.reshape(conv1, [-1]+conv1.get_shape().as_list()[1:])
print("conv1:%s"%conv1)
#k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
#conv1W = tf.Variable(net_data["conv1"][0])
#print("conv1W:%s"%conv1W)
#conv1b = tf.Variable(net_data["conv1"][1])
#print("conv1b:%s"%conv1b)
#conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
#conv1 = tf.nn.relu(conv1_in)
#print("conv1:%s"%conv1)

#第二层：最大池化层（55x55x96-->27x27x96)
pool1=tf.layers.max_pooling2d(inputs=lrn1, pool_size=[3,3], strides=2)
#pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

#第三层：卷积层(27x27x96->27x27x256)
#filters:256x96x96 strides:1 padding="SAME" 
#权重维度不对可能跟group有关，尝试将读取的权重参数合二为一
conv2W = tf.Variable(net_data["conv2"][0])
print("conv2W:%s"%conv2W)
conv2b = tf.Variable(net_data["conv2"][1])
print("conv2b:%s"%conv2b)
conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool1,conv2W,[1,1,1,1],padding="SAME"),conv2b))
conv2 = tf.reshape(conv2, [-1]+conv2.get_shape().as_list()[1:])
#k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
#conv2W = tf.Variable(net_data["conv2"][0])
#conv2b = tf.Variable(net_data["conv2"][1])
#conv2_in = conv(pool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
#conv2 = tf.nn.relu(conv2_in)

#第四层：最大池化层(27x27x256-->13x13x256)
pool2=tf.layers.max_pooling2d(inputs=lrn2, pool_size=[3, 3], strides=2)

#第五层：卷积层(13x13x256->13x13x384)
#filters:384x3x3 strides:1 padding="SAME" 
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool2,conv3W,[1,1,1,1],padding="SAME"),conv3b))
conv3=tf.reshape(conv3, [-1]+conv3.get_shape().as_list()[1:])

#第六层：卷积层(13x13x384->13x13x384)
#filters:384x3x3 strides:1 padding="SAME" 
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3,conv4W,[1,1,1,1],padding="SAME"),conv4b))
conv4 = tf.reshape(conv4, [-1]+conv4.get_shape().as_list()[1:])

#第七层：卷积层(13x13x384->13x13x256)
#filters:256x3x3 strides:1 padding="SAME" 
conv5W = tf.Variable(net_data["conv5"][0])
conv5b = tf.Variable(net_data["conv5"][1])
conv5=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4,conv5W,[1,1,1,1],padding="SAME"),conv5b))
conv5 = tf.reshape(conv5, [-1]+conv5.get_shape().as_list()[1:])

#第八层：最大池化层(13x13x256-->6x6x256)
pool3=tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)

#在输入全连接层前，先拍成一维向量(大小6x6x256=9216)
re1 = tf.reshape(pool2, [-1, 6 * 6 * 256])

#全连接层，只有全连接层才会进行L1和L2正则化
dense1 = tf.layers.dense(inputs=re1, 
                      units=4096, 
                      activation=tf.nn.relu,
		      		  use_bias=True,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
		      		  bias_initializer=tf.constant_initializer(0.1),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
dropout1 = tf.layers.dropout(
					 inputs=dense1,
					 rate=0.5,
					 )
dense2= tf.layers.dense(inputs=dropout1, 
                      units=4096, 
                      activation=tf.nn.relu,
		      		  use_bias=True,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
		      		  bias_initializer=tf.constant_initializer(0.1),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
dropout2 = tf.layers.dropout(
					 inputs=dense2,
					 rate=0.5,
					 )
logits= tf.layers.dense(inputs=dropout2, 
                        units=1000, 
                        activation=None,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
#---------------------------网络结束---------------------------
#SGD,batch_size=128,momentum=0.9,weight_decay=0.0005

loss=tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)    
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#开始训练
sess=tf.InteractiveSession()  
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

t = time.time()
output = sess.run(loss, feed_dict = {x:[im1,im2]})
################################################################################

#Output:


for input_im_ind in range(output.shape[0]):
    inds = argsort(output)[input_im_ind,:]
    print "Image", input_im_ind
    for i in range(5):
        print class_names[inds[-1-i]], output[input_im_ind, inds[-1-i]]

print time.time()-t
