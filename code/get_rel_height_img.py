import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt
def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

#first need to calculate the relative height in the image plane
img=cv2.imread('../result/5-lawn_result_mask.png')
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
left_cone_h_3=710.0-348.0
right_cone_h_3=835.0-460.0
rel_h_3=right_cone_h_3/left_cone_h_3
#4-bottle
left_cone_h_4=930.0-640.0
right_cone_h_4=715.0-168.0
rel_h_4=right_cone_h_4/left_cone_h_4
#5-lawn
left_cone_h_5=400.0-96.0
right_cone_h_5=450.0-344.0
rel_h_5=right_cone_h_5/left_cone_h_5