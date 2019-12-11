import cv2
import pickle
import numpy as np


haar_cascade = cv2.CascadeClassifier("venv/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yaml")

labels = {}
with open("labels.pickle", "rb") as f:
    old_labels = pickle.load(f)
    labels = {v:k for k,v in old_labels.items()}

capture = cv2.VideoCapture(0)

while (True):
    ret, frame = capture.read()
    #opencv's haar cascade can only read grayscale images, so we need to make one
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for x, y, w, h in faces:
        x_end_coord = x + w
        y_end_coord = y + h
        color = (0, 255, 0) #for some reason opencv uses bgr instead of rgb
        stroke = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)

        face_region_gray = gray[x: x_end_coord, y: y_end_coord]
        face_region = frame[x: x_end_coord, y: y_end_coord]
        cv2.rectangle(frame, (x, y), (x_end_coord, y_end_coord), color, stroke)

        id_, conf = recognizer.predict(face_region_gray)
        if conf >= 25:
            name = labels[id_]
        else:
            name = "unknown"

        cv2.putText(frame, name, (x,y), font, 1, text_color, stroke, cv2.LINE_AA)

        #img_item = "my-image.png"
        #cv2.imwrite(img_item, face_region_gray)

    cv2.imshow("frame", frame)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()

