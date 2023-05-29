import numpy as np
import cv2

imgOriginal = cv2.imread('tray3.jpg', cv2.IMREAD_COLOR)
imgBlur = cv2.medianBlur(imgOriginal,21)  
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(imgGray, 127, 255, 0)

ret, thresh = cv2.threshold(imgGray, 80, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

imax=0
areamax=0
for i in range(len(contours)):
    temp = contours[i]
    area = cv2.contourArea(temp)
    if area > areamax:
        imax=i
        areamax=area
tray = contours[imax]
area = cv2.contourArea(tray)
cv2.drawContours(imgOriginal, [tray], 0, (0,255,0), 3)
imgOriginal = cv2.putText(imgOriginal,"Area = "+str(areamax), (50,250),fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255,0,0), thickness=1)
imgRes = cv2.resize(imgOriginal, (0,0), fx=0.8, fy=0.8) 


circles = cv2.HoughCircles(imgGray, method=cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=100)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(imgOriginal,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(imgOriginal,(i[0],i[1]),2,(0,0,255),3)

BigCoinInTray=0
SmallCoinInTray=0
BigCoinOutTray=0
SmallCoinOutTray=0
for i in circles[0,:]:
    if cv2.pointPolygonTest(tray, (i[0], i[1]), False) > -1:
        if i[2] > 30  :
            BigCoinInTray += 1
        else:
            SmallCoinInTray += 1
    else:
        if i[2] > 30:
            BigCoinOutTray += 1
        else:
            SmallCoinOutTray += 1

imgOriginal = cv2.putText(imgOriginal, "BigCoinInTray = "+str(BigCoinInTray), (50,50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255,0,0), thickness=1)
imgOriginal = cv2.putText(imgOriginal, "BigCoinOutTray = "+str(BigCoinOutTray), (50,100), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255,0,0), thickness=1)
imgOriginal = cv2.putText(imgOriginal, "SmallCoinInTray = "+str(SmallCoinInTray), (50,150), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255,0,0), thickness=1)
imgOriginal = cv2.putText(imgOriginal, "SmallCoinOutTray = "+str(SmallCoinOutTray), (50,200), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255,0,0), thickness=1)
imgOrignal = cv2.putText(imgOriginal,"Area = "+str(areamax), (50,250),fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255,0,0), thickness=1)

cv2.imshow("Detected Circles", imgOriginal)
cv2.waitKey(0)
cv2.destroyAllWindows()
