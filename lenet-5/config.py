#coding:utf-8
#tensorflow 1.0.0
#是否训练
isTrain = True
#train vs. test ratio
ratio=0.8
#训练和测试数据，可将n_epoch设置更大一些
n_epoch=20
batch_size=64
#训练集数据
train_images_file = "./minist-database/train-images-idx3-ubyte"
train_labels_file = "./minist-database/train-labels-idx1-ubyte"
#测试集数据
valid_images_file = "./minist-database/t10k-images-idx3-ubyte"
valid_labels_file = "./minist-database/t10k-labels-idx1-ubyte"
#checkpoint
checkpoint_dir = './trained_model/' 
model_name = 'model-minist.ckpt'
