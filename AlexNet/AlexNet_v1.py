#encoding:utf-8
import tensorflow as tf
import numpy as np
from readImageNet import *
from config import *
#ImageNet数据集尺寸
w=224
h=224
c=3
#读取图片数据


#if isTrain:
#	arr=np.arange(nums)
#	#print(arr)
#	np.random.shuffle(arr)
#	print(arr[0])
#	data=data[arr]
#	label=label[arr]
#x_train = []
#y_train = []
#x_val = []
#y_val = []
##将所有数据分为训练集和验证集
##按照经验，8成为训练集，2成为验证集
#if isTrain:
#	s=np.int(nums*ratio)
#	#训练集
#	x_train=data[:s]
#	y_train=label[:s]
#	#验证集
#	x_val=data[s:]
#	y_val=label[s:]
#else:
#	x_train = data
#
#-----------------构建网络----------------------
#占位符
x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')

#第一层：卷积层（224x224x3-->55x55x96)
conv1=tf.layers.conv2d(
      inputs=x,
      filters=96,
	  strides=4,
      kernel_size=[11, 11],
      padding="valid",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
#第二层：最大池化层（55x55x96-->27x27x96)
pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[3,3], strides=2)

#第三层：卷积层(27x27x96->27x27x256)
conv2=tf.layers.conv2d(
      inputs=pool1,
      filters=256,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      use_bias=True,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      bias_initializer=tf.constant_initializer(0.0))
#第四层：最大池化层(27x27x256-->13x13x256)
pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

#第五层：卷积层(13x13x256->13x13x384)
conv3=tf.layers.conv2d(
      inputs=pool2,
      filters=384,
      kernel_size=[3,3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
#第六层：卷积层(13x13x384->13x13x384)
conv4=tf.layers.conv2d(
      inputs=conv3,
      filters=384,
      kernel_size=[3,3],
      padding="same",
      activation=tf.nn.relu,
      use_bias=True,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      bias_initializer=tf.constant_initializer(0.0))
#第七层：卷积层(13x13x384->13x13x256)
conv5=tf.layers.conv2d(
      inputs=conv4,
      filters=256,
      kernel_size=[3,3],
      padding="same",
      activation=tf.nn.relu,
      use_bias=True,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      bias_initializer=tf.constant_initializer(0.0))
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

#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=128, shuffle=False):
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


#开始训练
sess=tf.InteractiveSession()  
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

if isTrain:
	for epoch in range(n_epoch):
		#start_time = time.time()
		print("epoch %d"%epoch)
    		#training
		train_loss, train_acc, n_batch = 0, 0, 0
    		for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
			#print(x_train_a)
      			_,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
      			train_loss += err; train_acc += ac; n_batch += 1
    		print("   train loss: %f" % (train_loss/ n_batch))
    		print("   train acc: %f" % (train_acc/ n_batch))
    		print("\n")
    		if epoch == n_epoch - 1:
		 	print("Saving trained mode as ckpt format!")
		 	save_path = saver.save(sess,checkpoint_dir+model_name)
	    
		#validation
		val_loss, val_acc, n_batch = 0, 0, 0
		for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
			err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
	        	val_loss += err; val_acc += ac; n_batch += 1
		print("   validation loss: %f" % (val_loss/ n_batch))
		print("   validation acc: %f" % (val_acc/ n_batch))
		print("\n")
	    

#else:
#	class_begin_idx = 0
#	class_end_idx = 0
#	saver.restore(sess, checkpoint_dir+model_name)  
#	#print(label[0:20])
#	result = sess.run(logits,feed_dict={x:data})
#	class_index = np.argmax(result,axis=1)
#	#print(result)
#	#print(train_labels)
#        #class_result = []
#        correct_nums = 0
#        cnt_idx = 0
#	#print("The number is:\n")
#	#predict_result = []
#        for idx in class_index:
#                if idx == label[cnt_idx]:#softmax的标签为0开始的整数，所以这里数字0-9刚好对应了softmax的0-9层输出
#                	correct_nums += 1
#		#predict_result.append(idx)
#		
#                        #将错误图片显示出来
#                #else:
#                        #fig = plt.figure()
#                        #plt.imshow(data[class_begin_idx+error_idx ],cmap='binary')
#                        #plt.show()
#        	cnt_idx += 1
#	correct_pro = correct_nums / (len(result)*1.0)
#	#print(predict_result[0:20])
#	print("Input number of mnist set is %d, correct num:%d, correct proportion:%f"%(len(result), correct_nums,correct_pro))
#        print("\n")
sess.close()
