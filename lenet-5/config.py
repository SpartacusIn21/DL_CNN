#coding:utf-8
#是否训练
isTrain = False
#train vs. test ratio
ratio=0.8
#训练集数据
train_images_file = "./minist-database/train-images-idx3-ubyte"
train_labels_file = "./minist-database/train-labels-idx1-ubyte"
#测试集数据
valid_images_file = "./minist-database/t10k-images-idx3-ubyte"
valid_labels_file = "./minist-database/t10k-labels-idx1-ubyte"
#checkpoint
checkpoint_dir = './trained_model/' 
model_name = 'model-minist.ckpt'
