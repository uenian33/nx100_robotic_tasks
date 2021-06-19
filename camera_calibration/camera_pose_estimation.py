"""
This is the official example code provided by opencv 
"""
import numpy as np
import cv2 as cv
import glob

# Load previously saved data
mtx =  [[1176.96626,    0.,       727.86896],
         [   0.,      1178.14294,  518.23675],
         [   0.,         0.,         1.     ]]
dist =  [[1192.4447,     0.,       695.88598,],
         [   0.,      1193.82608,  518.85838,],
         [   0.,         0.,         1.     ]]
mtx_r =  [[ 0.10172,  0.03529 , 0.01247,  0.00165, -0.37752]]
dist_r =  [[ 0.12611, -0.09944 , 0.01237, -0.00181, -0.25298]]

# parameters defined for visualization
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

# Read the chessboard image to estimate camera's rotation matrix
vid = cv.VideoCapture(-1)
ret, img = vid.read()

# USing RAsanc for PNP pose estimation
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(gray, (7,6),None)
if ret == True:
    corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    # Find the rotation and translation vectors.
    ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
    # project 3D points to image plane
    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
    img = draw(img,corners2,imgpts)
    cv.imshow('img',img)
    k = cv.waitKey(0) & 0xFF
    if k == ord('s'):
        cv.imwrite(fname[:6]+'.png', img)

# rvecs is what we need, the rotation matrix/vector
print(rvecs)
