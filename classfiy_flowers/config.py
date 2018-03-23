#coding:utf-8
w=200
h=200
c=3
#是否训练
isTrain = False
#epoch,batch_size
n_epoch=20
batch_size=64
#train vs. test ratio
ratio=0.8
#训练集数据
#测试集数据
flower_path='./flower_photos/'
#checkpoint
checkpoint_dir = './trained_model/' 
model_name = 'model.ckpt'
