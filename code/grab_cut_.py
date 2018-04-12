import cv2
import numpy as np
from matplotlib import pyplot as plt


#person
rect_p=(190,62,260,380)

bgm=np.zeros((1,65),np.float64)
fgm=np.zeros((1,65),np.float64)

img_l=cv2.imread('../data_pair/5-lawn_left.jpg')
img_r=cv2.imread('../data_pair/5-lawn_left.jpg')
mask=np.zeros((img_l.shape[:2]),np.uint8)

print img_l.dtype

cv2.grabCut(img_l,mask,rect_p,bgm,fgm,5,cv2.GC_INIT_WITH_RECT)
mask_s=np.where((mask==2)|(mask==0),0,1).astype('uint8')
img_l_c=img_l*mask_s[:,:,np.newaxis]
cv2.imshow('gc',img_l_c)
cv2.waitKey(0)
fig=plt.figure()
plt.imshow(cv2.cvtColor(img_l,cv2.COLOR_BGR2RGB))
plt.show()
