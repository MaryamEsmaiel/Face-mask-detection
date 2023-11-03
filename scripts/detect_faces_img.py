import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def detect_face_img(input_img):

    img= np.array(input_img)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray_image.shape

    #Load classifier
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    #Perform the Face Detection

    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 50), 4)

    #Displaying the Image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_rgb
