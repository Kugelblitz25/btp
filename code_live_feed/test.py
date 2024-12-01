import numpy as np
import cv2
import time

#cap = cv2.VideoCapture("D:\\cctv_ncair.mp4")
#cap = cv2.VideoCapture("E:\Altum Tech AI Global Pvt Ltd\V-Analytics\mall.mp4")
#cap.open("rtsp://admin:transit@123@10.185.151.202:555/Streaming/channels/12/")
# cap = cv2.VideoCapture("rtsp://admin:transit@123@10.185.151.202/")
cap = cv2.VideoCapture("rtsp://ncair:ncair@10.185.153.173/cam/realmonitor?channel=2&subtype=0")



grab_i=[-15,-5,1,5,15,25,40]
wait_i=[150,75,25,1,1,1,1]
speed_i=2

point=1
cap.set(1,point)
ret, old_frame = cap.read()

while(True):
     # Capture frame-by-frame

    #for x in range(grab_i[speed_i]):
    #cap.grab()
    ret, frame = cap.read()
    if ret != 0:

        cv2.imshow('frame',frame)    

        key_pressed = cv2.waitKey(wait_i[speed_i])
        
        if key_pressed & 0xFF == ord('q'):
            break
        if (key_pressed & 0xFF == 62) or (key_pressed & 0xFF == 46) : # 62 > or 46 for .
            speed_i=min(speed_i+1,6)
        if (key_pressed & 0xFF == 60) or (key_pressed & 0xFF == 44): # 60 < or 44 for ,
            speed_i=max(speed_i-1,0)
        
    old_frame=frame
# When everything done, release the capture

cap.release()
cv2.destroyAllWindows()