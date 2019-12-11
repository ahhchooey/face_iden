import os
import cv2
import pickle
import numpy as np
from PIL import Image


haar_cascade = cv2.CascadeClassifier("venv/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
print(BASE_DIR)

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dir, files in os.walk(image_dir):
    #root is equal to os.path.dirname(path)
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()

            # we will turn the label into numbers
            # we will verify the image, and turn it into a numpy array and make it grayscale
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]
            pil_image = Image.open(path).convert("L") # converts image to grayscale
            image_array = np.array(pil_image, "uint8")

            faces = haar_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for x,y,w,h in faces:
                x_end_coord = x + w
                y_end_coord = y + h

                face_region = image_array[x: x_end_coord, y: y_end_coord]
                x_train.append(face_region)
                y_labels.append(id_)

with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yaml")
