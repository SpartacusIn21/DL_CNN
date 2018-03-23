# -*- coding: utf-8 -*-

import tensorflow as tf
from skimage import io,transform
import glob
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from config import *

#将所有的图片resize成100*100

classfication_dic_train = {0:"daisy", 1:"dandelion", 2:"roses", 3:"sunflowers", 4:"tulips"}#分类字典表
classfication_dic = {}#分类字典表
classfication_nums = {}#存储每个分类的数量
#读取图片
def read_img(path):
    cate=[path+'/'+x for x in os.listdir(path) if os.path.isdir(path+'/'+x)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
	nums = 0
	classfication_dic[idx] = folder.split('/')[-1]
	print(folder)
        for im in glob.glob(folder+'/*.jpg'):
            	print('reading the images:%s'%(im))
            	img=io.imread(im)
            	img=transform.resize(img,(w,h))
            	imgs.append(img)
            	labels.append(idx)
		nums += 1
	for key,val in classfication_dic.items():
		if val in folder:
			classfication_nums[val] = nums
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
data,label=read_img(flower_path)

print(classfication_dic)
print(classfication_nums)

#打乱顺序
num_example=data.shape[0]
if isTrain:
	arr=np.arange(num_example)
	np.random.shuffle(arr)
	data=data[arr]
	label=label[arr]

x_train = []
y_train = []
x_val = []
y_val = []

#将所有数据分为训练集和验证集
#按照经验，8成为训练集，2成为验证集
if isTrain:
	s=np.int(num_example*ratio)
	#训练集
	x_train=data[:s]
	y_train=label[:s]
	#验证集
	x_val=data[s:]
	y_val=label[s:]
else:
	x_train = data
	

#-----------------构建网络----------------------
#占位符
x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')

#第一个卷积层（200-->100)
conv1=tf.layers.conv2d(
      inputs=x,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

#第二个卷积层(100->50)
conv2=tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

#第三个卷积层(50->25)
conv3=tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

#第四个卷积层(25->12)
conv4=tf.layers.conv2d(
      inputs=pool3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

#在输入全连接层前，先拍成一维向量
re1 = tf.reshape(pool4, [-1, 12 * 12 * 128])

#全连接层，只有全连接层才会进行L1和L2正则化
dense1 = tf.layers.dense(inputs=re1, 
                      units=1024, 
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
dense2= tf.layers.dense(inputs=dense1, 
                      units=512, 
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
logits= tf.layers.dense(inputs=dense2, 
                        units=5, 
                        activation=None,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
#---------------------------网络结束---------------------------

loss=tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)    
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#定义一个函数，按批次取数据
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


#训练和测试数据，可将n_epoch设置更大一些

#训练 or 应用？
sess=tf.InteractiveSession()  
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

if isTrain:
	for epoch in range(n_epoch):
		start_time = time.time()

    		#training
		train_loss, train_acc, n_batch = 0, 0, 0
    		for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
      			_,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
      			train_loss += err; train_acc += ac; n_batch += 1
    		print("   train loss: %f" % (train_loss/ n_batch))
    		print("   train acc: %f" % (train_acc/ n_batch))
    		print("\n")
    		if epoch == n_epoch - 1:
		 	print("Saving trained mode as ckpt format!")
		 	save_path = saver.save(sess,checkpoint_dir+model_name )
	    
		#validation
		val_loss, val_acc, n_batch = 0, 0, 0
		for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
			err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
	        	val_loss += err; val_acc += ac; n_batch += 1
		print("   validation loss: %f" % (val_loss/ n_batch))
		print("   validation acc: %f" % (val_acc/ n_batch))
		print("\n")
	    

else:
	class_begin_idx = 0
	class_end_idx = 0
	saver.restore(sess, checkpoint_dir+model_name)  
	for key,val in classfication_dic.items():
		print(key)
		print(val)
		num = classfication_nums[val]
		class_end_idx += num 
		print("begin index:%d,end index:%d"%(class_begin_idx ,class_end_idx))
		result = sess.run(logits,feed_dict={x:x_train[class_begin_idx :class_end_idx ]})
		#print(result.shape)
		class_index = np.argmax(result,axis=1)
		print(class_index)
		#print(class_index)
		class_result = []
		correct_nums = 0
		error_idx = 0
		for idx in class_index:
			error_idx += 1
			if classfication_dic_train[idx] == val:
				correct_nums += 1
			#将错误图片显示出来
			#else:
				#fig = plt.figure()
				#plt.imshow(data[class_begin_idx+error_idx ],cmap='binary')
				#plt.show()
			class_result.append(classfication_dic_train[idx])
		class_begin_idx += num 
		print("Input flower is %s, input num:%d, correct num:%d"%(val, num,correct_nums))
		print(class_result)
		print("\n")

sess.close()
