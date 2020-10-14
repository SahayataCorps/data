import os
from pygazeanalyser.gazeplottervideo import draw_fixations, draw_heatmap, draw_scanpath, draw_raw
import numpy as np
import pandas as pd 
import cv2
from PIL import Image
import matplotlib.pyplot as plt



def graphs(fixData):
    name = input("Enter name by which you want to save outputs: ")
    #draw_scanpath(sasFix, sas, (1440,900), imagefile="bg-image.png", alpha=0.5, savefilename=str(name)+"_scanOutNew.png")
    draw_fixations(fixData, (1440,900), imagefile="bg-image.png", durationsize=True, durationcolour=False, alpha=0.5, savefilename=str(name)+"_fixOutNew.png")
    draw_heatmap(fixData, (1440,900), imagefile="bg-image.png", durationweight=True, alpha=0.5, savefilename=str(name)+"_heatOutNew.png")

# data = pd.read_csv("0_scanpath_data.csv")
# fixData = list(data.iloc[:,1:].values)
scanData = pd.read_csv("0_scanpath_data.csv")
sas = list(scanData.iloc[:,1:].values)
sasFix = list(scanData.iloc[:,1:6].values)
capt = cv2.VideoCapture("test.mp4")
out = cv2.VideoWriter("outScanpath.avi",cv2.VideoWriter_fourcc(*'XVID'),24,(1280,720))
i = 0
x = 0
durr=0
for x in  
while True:
    ret, frame = capt.read()
    if ret:
        row = []
        if i==0:
            row = [list(fixData[x])]
            durr = row[0][2]
            i+=1
        else:
            row = [list(fixData[x])]
            i+=1 
        draw_scanpath(sasFix, sas, (1440,900), imagefile="bg-image.png", alpha=0.5, savefilename=str(name)+"_scanOutNew.png")
        fig = draw_fixations(row, (1280,720), imagefile=None, durationsize=True, durationcolour=False, alpha=0.5)
        fig.canvas.draw ( )
 
        # Get the RGBA buffer from the figure
        w,h = fig.canvas.get_width_height()
        buf = np.fromstring ( fig.canvas.tostring_rgb(), dtype=np.uint8 )
        buf.shape = ( w, h,3)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll ( buf, 3, axis = 2 )
        w, h, d = buf.shape
        mask = Image.frombytes( "RGB", ( w ,h ), buf.tostring( ) )
        
        #print type(buf)
        print(i)
#        frame.save("outframes/"+str(x*10+i)+".jpg")
        
        out.write(np.array(mask))
        plt.close("all")
        if i==durr:
            x+=1
            i=0
            durr=0
    else:
        break




#draw_heatmap(fixData1+fixData2, (1440,900), imagefile="bg-image.png", durationweight=True, alpha=0.5, savefilename="avg_heatOutNew.png")