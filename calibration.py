import cv2 as cv
import numpy as np 
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('images/*.jpg')
for fname in images:
 img = cv.imread(fname)
 gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

 ret, corners = cv.findChessboardCorners(gray, (7,6), None)

 if ret == True:
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)
    img = cv.drawChessboardCorners(img, (7,6), corners2, ret)

 cv.imshow('img', img)
 cv.waitKey(500)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print('mtx')
print(mtx, '\n')
print('distortion')
print(dist)
np.savez("cam.npz",mtx,dist,rvecs,tvecs)

mean_error = 0
for i in range(len(objpoints)):
 imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
 error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
 mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )