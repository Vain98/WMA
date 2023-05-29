import cv2
import numpy as np

lower_red = np.array([0, 150, 150])
upper_red = np.array([5, 255, 255])

cap = cv2.VideoCapture('rgb_ball_720.mp4')

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        
        x, y, w, h = cv2.boundingRect(c)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
