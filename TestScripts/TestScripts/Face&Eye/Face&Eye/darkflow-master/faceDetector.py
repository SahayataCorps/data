import imutils
import dlib
import pandas as pd
import cv2
from darkflow.net.build import TFNet
import os
from os import listdir
from os.path import isfile, join

rpath = "rawDataset1"
wpath = "faces"
options = {
                "model" : "cfg/face_detection_tiny-yolo-voc-1c.cfg",
                "load" : 43125,
                "threshold" : 0.1,
                "gpu" : 0.95
                }
tfnet = TFNet(options)
dircs = next(os.walk(rpath))[1]

temp = {}
for x in dircs:
    mypath = rpath+"\\"+x
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    fileNo = 0
    #temp[x] = onlyfiles
    for y in onlyfiles:
        imagePath = mypath+"\\"+y
        frame = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        results = tfnet.return_predict(frame)
        #if len(results) == 0:
         #   rects = usingDlib(frame)
        for result in results:        
            label = result["label"]
            if label.lower()=='face':
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                face = frame[tl[1]:br[1], tl[0]:br[0]]
                cv2.imwrite(wpath+x+"\\"+y,face)      