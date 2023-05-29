import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('photo_1.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)


corners = cv.goodFeaturesToTrack(gray,maxCorners = 4, qualityLevel=0.01,minDistance=10,useHarrisDetector=True,k=0.04)
corners = np.int0(corners)
print(corners[0].ravel())
for i in corners:
    x,y = i.ravel()
    #ravel return a contiguous flattened array.
    cv.circle(img,(x,y),10,255,-1)
plt.imshow(img),plt.show()
cv.imwrite("02_goodFeaturesToTrack_img.jpg", img)

sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)
img=cv.drawKeypoints(gray,kp,img)
cv.imshow('photo_1.jpg',img)
cv.waitKey()
cv.destroyAllWindows()
cv.imwrite("photo_1.jpg", img)