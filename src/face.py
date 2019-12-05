import numpy as np
import cv2

haar_cascasde = cv2.CascadeClassifier("venv/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt.xml")

capture = cv2.VideoCapture(0)

while (True):
    ret, frame = capture.read()
    #opencv's haar cascade can only read grayscale images, so we need to make one
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = haar_cascasde.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for x, y, w, h in faces:
        x_end_coord = x + w
        y_end_coord = y + h
        color = (0, 255, 0) #for some reason opencv uses bgr instead of rgb
        stroke = 2

        face_region = frame[x: x_end_coord, y: y_end_coord]
        cv2.rectangle(frame, (x, y), (x_end_coord, y_end_coord), color, stroke)

    cv2.imshow("frame", frame)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
