#encoding:utf-8
#尝试利用toronto_edu_guerzhoy_tf_alexnet中的预训练参数来实现 单个GPU、无lrn处理的网络，失败告终，具体原因如后注释
#对比了每一个卷积后结果是否相等，发现在group=2的地方就不相等了，具体可参照图wrong_test中关于conv1计算的区别
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

##############################
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
		#print("input:%s,group:%s"%(input,group))
		#参照论文原文将数据和核函数分成两个GPU来计算
		print("conv!!!")
		print(input)
		input_groups = tf.split(input,group,3)
		print(input_groups)
		print(kernel)
		kernel_groups = tf.split(kernel, group,3)
		print(kernel_groups )
		output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
		#<tensorflow 1.0 version
        #conv = tf.concat(3, output_groups)
		#>=tensorflow 1.0 version
		#将计算结果再合并成起来
		conv = tf.concat(output_groups,3)
		#print(conv.get_shape().as_list())
		#print(conv.get_shape().as_list()[1:])
		#print(tf.nn.bias_add(conv, biases))
		#print([-1]+conv.get_shape().as_list()[1:])
		#print(tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:]))
		#reshape貌似没什么用
	#return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])
    return  tf.nn.bias_add(conv, biases)
##############################
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


#####################################
#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
#conv1W = tf.Variable(net_data["conv1"][0])
#print("conv1W:%s"%conv1W)
#conv1b = tf.Variable(net_data["conv1"][1])
#print("conv1b:%s"%conv1b)
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1_ = tf.nn.relu(conv1_in)
#####################################
temp_sess = tf.Session()
temp_sess.run(tf.global_variables_initializer())
print("conv1 equals:")
conv1_output= temp_sess.run(conv1, feed_dict = {x:[im1,im2]})
conv1__output= temp_sess.run(conv1_, feed_dict = {x:[im1,im2]})
print(np.array_equal(conv1_output,conv1__output))


#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
#####################################
#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1_ = tf.nn.local_response_normalization(conv1_,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
#####################################
temp_sess.run(tf.global_variables_initializer())
print("lrn1 equals:")
lrn1_output= temp_sess.run(lrn1 , feed_dict = {x:[im1,im2]})
lrn1__output= temp_sess.run(lrn1_, feed_dict = {x:[im1,im2]})
print(np.array_equal(lrn1_output,lrn1__output))

#第二层：最大池化层（55x55x96-->27x27x96)
#pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[3,3], strides=2)
pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
#####################################
#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
pool1_ = tf.nn.max_pool(lrn1_, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
#####################################
temp_sess.run(tf.global_variables_initializer())
print("pool1 equals:")
pool1_output= temp_sess.run(lrn1 , feed_dict = {x:[im1,im2]})
pool1__output= temp_sess.run(lrn1_, feed_dict = {x:[im1,im2]})
print(np.array_equal(pool1_output,pool1__output))

#第三层：卷积层(27x27x96->27x27x256)
#filters:256x96x96 strides:1 padding="SAME" 
#预训练参数conv2W 维度是针对原论文中两个GPU生成的,维度是256个5x5x48，需要重复一遍数据生成维度256个5x5x96核函数,conv2b维度是5x5x96
conv2W = tf.Variable(net_data["conv2"][0])
print(conv2W)
conv2W_ = tf.concat([conv2W,conv2W],2)
print(conv2W_)
#print("conv2W:%s"%conv2W)
conv2b = tf.Variable(net_data["conv2"][1])
#print("conv2b:%s"%conv2b)
conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool1,conv2W_ ,[1,1,1,1],padding="SAME"),conv2b))
#print("output/input:%s"%conv2)

#####################################
#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2_in = conv(pool1_, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2_ = tf.nn.relu(conv2_in)
#####################################
temp_sess.run(tf.global_variables_initializer())
#因为kenel尺寸是256个5x5x48，直接重复合并成256个5x5x96,跟输入5x5x96做卷积的结果并不等于拆分成上下两个128x5x5x48和上下两个28x28x48分别卷积后再concat的值
print("conv2 equals:")
conv2_output= temp_sess.run(conv2, feed_dict = {x:[im1,im2]})
conv2__output= temp_sess.run(conv2_, feed_dict = {x:[im1,im2]})
print(np.array_equal(conv2_output,conv2__output))

#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#第四层：最大池化层(27x27x256-->13x13x256)
#pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)
pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

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
#saver = tf.train.Saver()
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
