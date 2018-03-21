import numpy as np
from skimage import io,transform
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
