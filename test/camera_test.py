import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    #capturing frames
    ret, frame = cap.read()

    #showing the frames as video
    cv2.imshow("frame", frame)

    #cancel feed with q key
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
