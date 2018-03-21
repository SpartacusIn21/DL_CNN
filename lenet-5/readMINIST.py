#encoding:utf-8
import numpy as np
import struct
import matplotlib.pyplot as plt
from skimage import transform
#读取minist images数据集
def read_minist_image_data(file_name):
	file = open(file_name,'rb')
	buff = file.read()#一次性读取全部数据
	header_format = '>IIII'
	#偏移量
	buf_pointer = 0
	#图片大小尺寸信息
	magic_number=0
	number_of_images = 0
	number_of_rows = 0
	number_of_columns = 0
	#读取images数据
	magic_number,number_of_images ,number_of_rows ,number_of_columns = struct.unpack_from(header_format,buff,buf_pointer)
	print("magic_number:%d,number_of_images:%d,number_of_rows:%d,number_of_columns:%d"%(magic_number,number_of_images,number_of_rows,number_of_columns))
	buf_pointer += struct.calcsize(header_format)
	images = []#图片数据
	image_byte_size = number_of_rows * number_of_columns 
	date_format = ">%dB"%image_byte_size 
	for idx in range(number_of_images):
	#for idx in range(1):
		image = struct.unpack_from(date_format ,buff,buf_pointer)
		#print(image)
		#偏移
		buf_pointer += struct.calcsize(date_format )
		image = np.array(image)
		#print(image.shape)
		#print(image)
		#image = transform.resize(image,(number_of_rows,number_of_columns))
		image = image.reshape(number_of_rows, number_of_columns,1)
		#print(image.shape)
		#print(image)
		#显示图片
		#fig = plt.figure()
		#plt.imshow(image,cmap='binary')
		#plt.show()
		images.append(image)
	file.close()
	#返回图片数据
	return number_of_images ,number_of_rows ,number_of_columns,np.asarray(images,np.ubyte)
#读取minist label数据集
def read_minist_label_data(file_name):
	file = open(file_name,'rb')
	buff = file.read()#一次性读取全部数据
	header_format = ''	
	#偏移量
	buf_pointer = 0
	#图片大小尺寸信息
	magic_number=0
	number_of_labels = 0
	#读取label数据
	header_format = '>II'
	magic_number,number_of_labels = struct.unpack_from(header_format,buff,buf_pointer)
	print("magic_number:%d,number_of_labels:%d"%(magic_number,number_of_labels))
	buf_pointer += struct.calcsize(header_format)
	labels = []#label数据
	date_format = ">B"
	for idx in range(number_of_labels):
	#for idx in range(10):
		label = struct.unpack_from(date_format ,buff,buf_pointer)
		#偏移
		buf_pointer += struct.calcsize(date_format )
		#输出label(因为label输出是tuple)
		#print(label[0])
		labels.append(label[0])
	file.close()
	return number_of_labels ,np.asarray(labels,np.ubyte)

