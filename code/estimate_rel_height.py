import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt
def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))


def cal_rel_height(file_name,rel_h_img):
    # calculate disparity
    disp_map = cv2.imread('../result/'+file_name+'_result_mask.png')
    # cv2.imshow('disp',disp_map)
    # cv2.waitKey(0)
    dens_disp_map = cv2.fastNlMeansDenoisingColored(disp_map)
    # cv2.imshow('disp',dens_disp_map)
    # cv2.waitKey(0)
    # print  dens_disp_map.shape
    if file_name=='1-cones':
        rect_left = (110, 260, 90, 150)
        rect_right = (206, 250, 80, 130)
    if file_name=='2-book':
        rect_left = (380, 500, 260, 380)
        rect_right = (650, 400, 320, 400)
    if file_name == '3-box':
        rect_left = (270, 290, 360, 430)
        rect_right = (742, 474, 500, 410)
    if file_name=='4-bottle':
        rect_left = (250, 700, 300, 240)
        rect_right = (580, 170, 400, 590)
    if file_name=='5-lawn':
        rect_left = (235, 70, 190, 300)
        rect_right = (765, 375, 180, 140)

    mask = np.zeros((dens_disp_map.shape[:2]), np.uint8)
    bgm = np.zeros((1, 65), np.float64)
    fgm = np.zeros((1, 65), np.float64)
    cv2.grabCut(dens_disp_map, mask, rect_left, bgm, fgm, 5, cv2.GC_INIT_WITH_RECT)
    mask_s = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    disp_map_left = dens_disp_map * mask_s[:, :, np.newaxis]

    fig_1 = plt.figure()
    plt.imshow(cv2.cvtColor(disp_map_left, cv2.COLOR_BGR2RGB))
    # plt.show()


    mask = np.zeros((dens_disp_map.shape[:2]), np.uint8)
    bgm = np.zeros((1, 65), np.float64)
    fgm = np.zeros((1, 65), np.float64)
    cv2.grabCut(dens_disp_map, mask, rect_right, bgm, fgm, 5, cv2.GC_INIT_WITH_RECT)
    mask_s = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    disp_map_right = dens_disp_map * mask_s[:, :, np.newaxis]
    fig_2 = plt.figure()
    plt.imshow(cv2.cvtColor(disp_map_right, cv2.COLOR_BGR2RGB))
    plt.show()

    height = disp_map_left.shape[0]
    width = disp_map_left.shape[1]

    # calculate average disparity value
    count_l = 0
    disp_val_left = 0.0
    for i in range(height):
        for j in range(width):
            if disp_map_left[i, j, 0] != 0:
                # print disp_map_left[i,j,0]
                count_l += 1
                disp_val_left += disp_map_left[i, j, 0]
    print count_l
    disp_val_left = disp_val_left / float(count_l)
    print disp_val_left

    count_r = 0
    disp_val_right = 0.0
    for i in range(height):
        for j in range(width):
            if disp_map_right[i, j, 0] != 0:
                # print disp_map_left[i,j,0]
                count_r += 1
                disp_val_right += disp_map_right[i, j, 0]
    print count_r
    disp_val_right = disp_val_right / float(count_r)
    print disp_val_right

    h_r_div_h_l = (disp_val_right / disp_val_left) * rel_h_img
    return h_r_div_h_l



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



print cal_rel_height('5-lawn',rel_h_5)


