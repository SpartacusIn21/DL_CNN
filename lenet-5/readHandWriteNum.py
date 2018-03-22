import numpy as np
from skimage import io,transform
def read_img(path,w,h):
    cate=[path+'/'+x for x in os.listdir(path) if os.path.isdir(path+'/'+x)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            	print('reading the images:%s'%(im))
            	img=io.imread(im)
            	img=transform.resize(img,(w,h))
            	imgs.append(img)
            	labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
