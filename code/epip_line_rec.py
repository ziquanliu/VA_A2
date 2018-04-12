import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt

def drawlines(img_L,img_R,lines,pts1,pts2):
    c = img_L.shape[1]
    img_L_g = cv2.cvtColor(img_L,cv2.COLOR_GRAY2BGR)
    img_R_g = cv2.cvtColor(img_R,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        #print x0,y0
        cv2.line(img_L_g, (x0,y0), (x1,y1), (0, 0, 255),2)
        cv2.circle(img_L_g,tuple(pt1),5,(255, 0, 0),-2)
        cv2.circle(img_R_g,tuple(pt2),5,(255, 0, 0),-2)
    return img_L_g,img_R_g

def plot_epip_rec(file_name):
    img_L = cv2.imread('../data_pair/'+file_name+'_left.jpg', 0)
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
    for i in range(n_match):
        arr_pL[i, :] = point_L[i]
        arr_pR[i, :] = point_R[i]



    # print Four_pL
    # print Four_pR
    height = img_L.shape[0]
    wid = img_L.shape[1]
    F, mask = cv2.findFundamentalMat(np.float32(arr_pL), np.float32(arr_pR), cv2.FM_RANSAC)
    # print mask
    retval, H_L, H_R = cv2.stereoRectifyUncalibrated(arr_pL.reshape(-1, 1), arr_pR.reshape(-1, 1), F, (wid, height))

    # print retval
    # print H_L





    inl_pL = np.float32(arr_pL[mask.ravel() == 1])
    inl_pR = np.float32(arr_pR[mask.ravel() == 1])


    # print inl_pL
    # print inl_pR
    # print inl_pL
    # print inl_pR
    # calculate the left epipolar line
    line_L = cv2.computeCorrespondEpilines(inl_pR, 2, F)
    line_R = cv2.computeCorrespondEpilines(inl_pL, 1, F)

    point_draw = np.random.choice(inl_pL.shape[0], 4)
    Four_pL = np.zeros((4, 2),dtype='uint32')
    Four_line_L=np.zeros((4,1,3))
    Four_pR = np.zeros((4, 2),dtype='uint32')
    Four_line_R=np.zeros((4,1,3))
    for i in range(4):
        Four_pR[i,:]=inl_pR[point_draw[i],:]
        Four_pL[i, :] = inl_pL[point_draw[i], :]
        Four_line_L[i,:,:]=line_L[point_draw[i],:,:]
        Four_line_R[i, :, :] = line_R[point_draw[i], :, :]
    #print Four_line_L.shape
    Four_line_L = Four_line_L.reshape(-1, 3)
    Four_line_R = Four_line_R.reshape(-1, 3)

    # print line_L
    img_L_lep, img_R_lep = drawlines(img_L, img_R, Four_line_L, Four_pL, Four_pR)
    # calculate the right epipolar line
    # print line_R

    #print Four_pL
    #print Four_pR
    #print inl_pR[0,:]
    img_R_rep, img_L_rep = drawlines(img_R, img_L, Four_line_R, Four_pR, Four_pL)

    rect_L_w_ep = cv2.warpPerspective(img_L_lep, H_L, (wid, height))
    rect_R_w_ep = cv2.warpPerspective(img_R_rep, H_R, (wid, height))

    rect_L = cv2.warpPerspective(img_L, H_L, (wid, height))
    rect_R = cv2.warpPerspective(img_R, H_R, (wid, height))

    return rect_L_w_ep,rect_R_w_ep


    #return rect_L_w_ep,rect_R_w_ep
    #cv2.imshow('rec left', rect_L_w_ep)
    #cv2.imshow('rec right', rect_R_w_ep)
    #cv2.waitKey(0)


if __name__ == "__main__":
    file_name='1-cones'
    img_l_1,img_r_1=plot_epip_rec(file_name)
    file_name='2-book'
    img_l_2,img_r_2=plot_epip_rec(file_name)
    file_name='3-box'
    img_l_3,img_r_3=plot_epip_rec(file_name)
    file_name='4-bottle'
    img_l_4,img_r_4=plot_epip_rec(file_name)
    file_name='5-lawn'
    img_l_5,img_r_5=plot_epip_rec(file_name)



    fig = plt.figure(figsize=(15,40))
    plt.subplot(5, 2, 1)
    plt.imshow(cv2.cvtColor(img_l_1, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('left')
    plt.subplot(5, 2, 2)
    plt.imshow(cv2.cvtColor(img_r_1, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('right')

    plt.subplot(5, 2, 3)
    plt.imshow(cv2.cvtColor(img_l_2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(5, 2, 4)
    plt.imshow(cv2.cvtColor(img_r_2, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(5, 2, 5)
    plt.imshow(cv2.cvtColor(img_l_3, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(5, 2, 6)
    plt.imshow(cv2.cvtColor(img_r_3, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(5, 2, 7)
    plt.imshow(cv2.cvtColor(img_l_4, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(5, 2, 8)
    plt.imshow(cv2.cvtColor(img_r_4, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(5, 2, 9)
    plt.imshow(cv2.cvtColor(img_l_5, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(5, 2, 10)
    plt.imshow(cv2.cvtColor(img_r_5, cv2.COLOR_BGR2RGB))
    plt.axis('off')


    plt.savefig('../result/rec_epi.eps',dpi=150)
    #plt.show()

