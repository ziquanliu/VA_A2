import cv2
import numpy as np
from matplotlib import pyplot as plt

def drawlines(img_L,img_R,lines,pts1,pts2):
    c = img_L.shape[1]
    img_L_g = cv2.cvtColor(img_L,cv2.COLOR_GRAY2BGR)
    img_R_g = cv2.cvtColor(img_R,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        #print x0,y0
        cv2.line(img_L_g, (x0,y0), (x1,y1), color,1)
        cv2.circle(img_L_g,tuple(pt1),5,color,-1)
        cv2.circle(img_R_g,tuple(pt2),5,color,-1)
    return img_L_g,img_R_g



img_L=cv2.imread('../data_pair/2-book_left.jpg',0)
img_R=cv2.imread('../data_pair/2-book_right.jpg',0)

sift=cv2.SIFT()

kp_L,des_L=sift.detectAndCompute(img_L,None)
kp_R,des_R=sift.detectAndCompute(img_R,None)


FLANN_INDEX_KDTREE = 0

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

search_params = dict(checks=100)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_L, des_R, k=2)

good_mat=[]
point_L=[]
point_R=[]

for i,(k_1,k_2) in enumerate(matches):
    if k_1.distance<0.8*k_2.distance:
        good_mat.append(k_1)
        point_L.append(kp_L[k_1.queryIdx].pt)
        point_R.append(kp_R[k_1.trainIdx].pt)
n_match=len(good_mat)
arr_pL=np.zeros((n_match,2))
arr_pR=np.zeros((n_match,2))
for i in range(n_match):
    arr_pL[i,:]=point_L[i]
    arr_pR[i,:]=point_R[i]


F,mask=cv2.findFundamentalMat(np.float32(arr_pL),np.float32(arr_pR),cv2.FM_RANSAC)
retval,H_L,H_R=cv2.stereoRectifyUncalibrated(arr_pL.reshape(-1,1),arr_pR.reshape(-1,1),F,img_L.shape)

print retval


inl_pL=np.float32(arr_pL[mask.ravel()==1])
inl_pR=np.float32(arr_pR[mask.ravel()==1])
#print inl_pL
#print inl_pR
#calculate the left epipolar line
line_L=cv2.computeCorrespondEpilines(inl_pR,2,F)
line_L=line_L.reshape(-1,3)
#print line_L
img_L_lep,img_R_lep=drawlines(img_L,img_R,line_L,inl_pL,inl_pR)
#calculate the right epipolar line
line_R=cv2.computeCorrespondEpilines(inl_pL,1,F)
line_R=line_R.reshape(-1,3)
#print line_R
img_R_rep,img_L_rep=drawlines(img_R,img_L,line_R,inl_pR,inl_pL)


#cv2.imshow('left epipolar line',img_L_lep)

#cv2.imshow('right epipolar line',img_R_rep)
#cv2.waitKey(0)