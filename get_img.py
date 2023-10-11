import cv2 as cv

cap = cv.VideoCapture(0)

ctr = 0

while cap.isOpened():

    ret,frame = cap.read()

    k = cv.waitKey(5)

    if k==27:
        break
    elif k==ord('s'):
        cv.imwrite('images/img'+str(ctr)+'.jpg',frame)
        print('images/img'+str(ctr)+'.jpg'+' saved')
        ctr+=1
    
    cv.imshow('Image',frame)

cap.release()
cv2.destroyAllWindows()

