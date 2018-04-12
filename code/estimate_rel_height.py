import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt
def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

#first need to calculate the relative height in the image plane
img=cv2.imread('../data_pair/2-book_left.jpg')
fig=plt.figure()
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()
cid=fig.canvas.mpl_connect('button press event',onclick)

#1-cone
left_cone_h_1=411.0-268.0
right_cone_h_1=374.0-261.0
rel_h_1=right_cone_h_1/left_cone_h_1
#2-book
left_cone_h_2=887.0-522.0
right_cone_h_2=811.0-490.0
rel_h_2=right_cone_h_2/left_cone_h_2
#3-box

#4-bottle

#5-lawn


#calculate disparity
disp_map=cv2.imread('../result/2-book_result_mask.png')
#cv2.imshow('disp',disp_map)
#cv2.waitKey(0)
dens_disp_map=cv2.fastNlMeansDenoisingColored(disp_map)
#cv2.imshow('disp',dens_disp_map)
#cv2.waitKey(0)
#print  dens_disp_map.shape

mask=np.zeros((dens_disp_map.shape[:2]),np.uint8)
rect_left=(400,500,300,400)
bgm=np.zeros((1,65),np.float64)
fgm=np.zeros((1,65),np.float64)
cv2.grabCut(dens_disp_map,mask,rect_left,bgm,fgm,5,cv2.GC_INIT_WITH_RECT)
mask_s=np.where((mask==2)|(mask==0),0,1).astype('uint8')
disp_map_left=dens_disp_map*mask_s[:,:,np.newaxis]


fig_1=plt.figure()
plt.imshow(cv2.cvtColor(disp_map_left,cv2.COLOR_BGR2RGB))
#plt.show()


mask=np.zeros((dens_disp_map.shape[:2]),np.uint8)
rect_right=(700,450,300,380)
bgm=np.zeros((1,65),np.float64)
fgm=np.zeros((1,65),np.float64)
cv2.grabCut(dens_disp_map,mask,rect_right,bgm,fgm,5,cv2.GC_INIT_WITH_RECT)
mask_s=np.where((mask==2)|(mask==0),0,1).astype('uint8')
disp_map_right=dens_disp_map*mask_s[:,:,np.newaxis]
fig_2=plt.figure()
plt.imshow(cv2.cvtColor(disp_map_right,cv2.COLOR_BGR2RGB))
plt.show()

height=disp_map_left.shape[0]
width=disp_map_left.shape[1]


#calculate average disparity value
count_l=0
disp_val_left=0.0
for i in range(height):
    for j in range(width):
        if disp_map_left[i,j,0]!=0:
            #print disp_map_left[i,j,0]
            count_l+=1
            disp_val_left+=disp_map_left[i,j,0]
print count_l
disp_val_left=disp_val_left/float(count_l)
print disp_val_left


count_r=0
disp_val_right=0.0
for i in range(height):
    for j in range(width):
        if disp_map_right[i,j,0]!=0:
            #print disp_map_left[i,j,0]
            count_r+=1
            disp_val_right+=disp_map_right[i,j,0]
print count_r
disp_val_right=disp_val_right/float(count_r)
print disp_val_right


h_r_div_h_l=(disp_val_right/disp_val_left)*rel_h_2
print h_r_div_h_l
