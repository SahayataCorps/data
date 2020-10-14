import imutils
import dlib
import pandas as pd
import cv2
from darkflow.net.build import TFNet
import os
from os import listdir
from os.path import isfile, join



options2 = {
                "model" : "cfg/yolov2-2c.cfg",
                "load" : 1075,
                "threshold" : 0.25,
                "gpu" : 1.0
                }
eyes = TFNet(options2)
rpath = "outputs"
wpath = "outputs"
dircs = next(os.walk(rpath))[1]
print(dircs)
temp = {}
for x in dircs:
    mypath = rpath+"\\"+x
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    fileNo = 0
    #temp[x] = onlyfiles
    for y in onlyfiles:
        imagePath = mypath+"\\"+y
        print(imagePath)
        frame = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        results = eyes.return_predict(frame)
        #if len(results) == 0:
         #   rects = usingDlib(frame)
        for result in results:        
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            frame = cv2.rectangle(frame, tl, br, (0,255,0),3)

        cv2.imwrite(wpath+"\\"+y,frame)      








