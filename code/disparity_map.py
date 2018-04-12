import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt




def cal_rectit_mat(file_name):
    img_L = cv2.imread('../data_pair/'+file_name+'_left.jpg', 0)
    print img_L.dtype
    img_R = cv2.imread('../data_pair/'+file_name+'_right.jpg', 0)

    sift = cv2.SIFT()

    kp_L, des_L = sift.detectAndCompute(img_L, None)
    kp_R, des_R = sift.detectAndCompute(img_R, None)

    FLANN_INDEX_KDTREE = 0

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_L, des_R, k=2)

    good_mat = []
    point_L = []
    point_R = []

    for i, (k_1, k_2) in enumerate(matches):
        if k_1.distance < 0.8 * k_2.distance:
            good_mat.append(k_1)
            point_L.append(kp_L[k_1.queryIdx].pt)
            point_R.append(kp_R[k_1.trainIdx].pt)
    n_match = len(good_mat)
    arr_pL = np.zeros((n_match, 2))
    arr_pR = np.zeros((n_match, 2))
    # print n_match
    for i in range(n_match):
        arr_pL[i, :] = point_L[i]
        arr_pR[i, :] = point_R[i]
    num_point = n_match
    point_ch = np.random.choice(arr_pL.shape[0], num_point)
    # print point_ch
    Four_pL = np.zeros((num_point, 2))
    Four_pR = np.zeros((num_point, 2))
    for i in range(num_point):
        Four_pL[i, :] = arr_pL[point_ch[i], :]
        Four_pR[i, :] = arr_pR[point_ch[i], :]

    # print Four_pL
    # print Four_pR
    height = img_L.shape[0]
    wid = img_L.shape[1]
    F, mask = cv2.findFundamentalMat(np.float32(Four_pL), np.float32(Four_pR), cv2.FM_RANSAC)
    # print mask
    retval, H_L, H_R = cv2.stereoRectifyUncalibrated(Four_pL.reshape(-1, 1), Four_pR.reshape(-1, 1), F, (wid, height))

    # print inl_pL
    # print inl_pR
    # print inl_pL
    # print inl_pR
    # calculate the left epipolar line
    # print line_L
    # calculate the right epipolar line
    # print line_R


    rect_L = cv2.warpPerspective(img_L, H_L, (wid, height))
    rect_R = cv2.warpPerspective(img_R, H_R, (wid, height))

    minDis = 0
    numDis = 256 - minDis
    SADWS = 10
    P1 = 8 * 1 * SADWS * SADWS
    P2 = 32 * 1 * SADWS * SADWS

    stereo = cv2.StereoSGBM(minDisparity=minDis, numDisparities=numDis, SADWindowSize=SADWS, P1=P1, P2=P2,
                            disp12MaxDiff=64, preFilterCap=1, uniquenessRatio=5, speckleWindowSize=0, speckleRange=20,
                            fullDP=False)
    disp = stereo.compute(rect_L, rect_R).astype(np.float32)

    norm_disp = (disp - minDis) / numDis
    #print norm_disp
    #cv2.imshow('disparity map', norm_disp)
    #cv2.waitKey(0)
    return H_L,H_R,norm_disp

def get_disp_map(file_name,mask_name):
    H_L, H_R, norm_disp = cal_rectit_mat(file_name)
    mask_s = pickle.load(open('../data_pair/' + str(mask_name) + '_l_mask.txt', 'rb'))
    height = norm_disp.shape[0]
    wid = norm_disp.shape[1]
    # print height
    # print wid

    rect_mask = cv2.warpPerspective(mask_s, H_L, (wid, height))
    # print rect_mask.shape
    # print norm_disp.shape


    mask_disp = cv2.bitwise_and(norm_disp, norm_disp, mask=rect_mask)
    return norm_disp,mask_disp



file_name='1-cones'
mask_name=1
norm_disp_1,mask_disp_1=get_disp_map(file_name,mask_name)
file_name='2-book'
mask_name=2
norm_disp_2,mask_disp_2=get_disp_map(file_name,mask_name)
file_name='3-box'
mask_name=3
norm_disp_3,mask_disp_3=get_disp_map(file_name,mask_name)
file_name='4-bottle'
mask_name=4
norm_disp_4,mask_disp_4=get_disp_map(file_name,mask_name)
file_name='5-lawn'
mask_name=5
norm_disp_5,mask_disp_5=get_disp_map(file_name,mask_name)

fig = plt.figure(figsize=(15, 40))
plt.subplot(5, 2, 1)
plt.imshow(cv2.cvtColor(norm_disp_1, cv2.COLOR_GRAY2RGB))
plt.axis('off')
plt.title('image disparity')
plt.subplot(5, 2, 2)
plt.imshow(cv2.cvtColor(mask_disp_1, cv2.COLOR_GRAY2RGB))
plt.axis('off')
plt.title('object disparity')

plt.subplot(5, 2, 3)
plt.imshow(cv2.cvtColor(norm_disp_2, cv2.COLOR_GRAY2RGB))
plt.axis('off')
plt.subplot(5, 2, 4)
plt.imshow(cv2.cvtColor(mask_disp_2, cv2.COLOR_GRAY2RGB))
plt.axis('off')

plt.subplot(5, 2, 5)
plt.imshow(cv2.cvtColor(norm_disp_3, cv2.COLOR_GRAY2RGB))
plt.axis('off')
plt.subplot(5, 2, 6)
plt.imshow(cv2.cvtColor(mask_disp_3, cv2.COLOR_GRAY2RGB))
plt.axis('off')

plt.subplot(5, 2, 7)
plt.imshow(cv2.cvtColor(norm_disp_4, cv2.COLOR_GRAY2RGB))
plt.axis('off')
plt.subplot(5, 2, 8)
plt.imshow(cv2.cvtColor(mask_disp_4, cv2.COLOR_GRAY2RGB))
plt.axis('off')

plt.subplot(5, 2, 9)
plt.imshow(cv2.cvtColor(norm_disp_5, cv2.COLOR_GRAY2RGB))
plt.axis('off')
plt.subplot(5, 2, 10)
plt.imshow(cv2.cvtColor(mask_disp_5, cv2.COLOR_GRAY2RGB))
plt.axis('off')

plt.savefig('../result/disparity_map.eps', dpi=80)



