"""
This script is to test eye model created using yolo.
Seperate face model and eye model.
Face detector used is the one created using yolo.
"""
import imutils
import dlib
import pandas as pd
import cv2
from darkflow.net.build import TFNet
import os
from os import listdir
from os.path import isfile, join
import dlib

options1 = {
                "model" : "cfg/face_detection_tiny-yolo-voc-1c.cfg",
                "load" : 43125,
                "threshold" : 0.1,
                "gpu" : 0
                }
faces = TFNet(options1)

options2 = {
                "model" : "cfg/tiny-yolo-voc-1c.cfg",
                "load" : 270,
                "threshold" : 0.5,
                "gpu" : 0
                }
eyes = TFNet(options2)
#detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        results = faces.return_predict(frame)
        for result in results:        
            label = result["label"]
            if label.lower()=='eyes':
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                face = frame[tl[1]:br[1], tl[0]:br[0]]
                res2 = eyes.return_predict(face)
                for result in res2:        
                    label = result["label"]
                    if label.lower()=='eyes':
                        tl2 = (result['topleft']['x'], result['topleft']['y'])
                        br2 = (result['bottomright']['x'], result['bottomright']['y'])
                        face = cv2.rectangle(face, tl2, br2, (0,0,255),3)
            cv2.imshow("output",face)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
cap.release()
cv2.destroyAllWindows()
