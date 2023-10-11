import cv2 as cv
import numpy as np 
import glob

def draw(img,corner,imgpt):
    imgpt = np.int32(imgpts).reshape(-1,2)
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpt[i]), tuple(imgpt[j]),(0,0,255),3)
        img = cv.drawContours(img, [imgpt[4:]],-1,(0,0,255),3)
    return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
axis =  np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])


with np.load('cam.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('arr_0','arr_1','arr_2','arr_3')]

for frame in glob.glob('images/*.jpg'):
    img = cv.imread(frame)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)

    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        ret, rvecs,tvecs = cv.solvePnP(objp,corners2,mtx,dist)
        imgpts,jac = cv.projectPoints(axis,rvecs,tvecs,mtx,dist)
        img = draw(img,corners2,imgpts)
    
    cv.imshow('img',img)
    k = cv.waitKey(0) & 0xFF #and is taken to take only the first 8 bytes. NumLock can mess up the value of input k 
    if ord('k') == k:
        cv.imwrite(frame[:3]+'.jpg',img)

cv.destroyAllWindows()

 
