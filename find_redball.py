import cv2
import sys
import numpy as np

img = cv2.imread("red_ball.jpg")
if img is None:
    sys.exit("Image not found!")


hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
h,s,v=cv2.split(hsv)
hsv_img = cv2.merge((h,s,v))


lower_bounds = np.array([100, 90, 50])
upper_bounds = np.array([190, 255, 255])

mask = cv2.inRange(hsv, lower_bounds, upper_bounds)
cv2.imshow("mask", mask)


kernel = np.ones((7,7),np.uint8)

mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
cv2.imshow("Improved Image", mask)

segment_img = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("segment", segment_img)

contors, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = cv2.drawContours(img, contors, -1, (1, 1, 255), 3)
cv2.imshow("output", output)

gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

for i in contors:
    M = cv2.moments(i)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(img, (cx, cy), 3, (255,255,255), -1)
        cv2.putText(img, "red ball", (cx - 20, cy - 20), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,0), 2)
cv2.imshow("centroid", img)

cv2.waitKey()




