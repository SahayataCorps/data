"""This is a test script to test pre-built 5 points eye landmarks""" 
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import pandas as pd

detector = dlib.get_frontal_face_detector()
#cnn_face_detection_model_v1
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
cap = cv2.VideoCapture(0)
while True:
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            box = face_utils.rect_to_bb(rect)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        cv2.imshow("output",image)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    else:
        break
cap.release()
cv2.destroyAllWindows()
