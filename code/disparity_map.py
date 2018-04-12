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
    SADWS = 18
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


H_L,H_R,norm_disp=cal_rectit_mat('2-book')
mask_s=pickle.load(open('../data_pair/2_l_mask.txt','rb'))
height=norm_disp.shape[0]
wid=norm_disp.shape[1]
#print height
#print wid

rect_mask = cv2.warpPerspective(mask_s, H_L, (wid, height))
#print rect_mask.shape
#print norm_disp.shape


mask_disp=cv2.bitwise_and(norm_disp,norm_disp,mask=rect_mask)
#print mask_disp[350,:]
print mask_disp.dtype
#print norm_disp.dtype
int_mask_disp=(np.round(mask_disp*255.0)).astype(np.uint8)
cv2.imwrite('../result/2-book_result_mask.png',cv2.cvtColor(int_mask_disp,cv2.COLOR_GRAY2RGB))
#print rgb_mask_disp[350,:,1]


fig_1=plt.figure()
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(norm_disp,cv2.COLOR_GRAY2RGB))
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(mask_disp,cv2.COLOR_GRAY2RGB))
plt.show()


