"""
This is a test file to create multi-user heatmap on video
"""
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

#reading multiple data
completeData = []
for x in range(20):
    data = pd.read_csv(str(x)+"_fixation_data.csv")
    fixData = list(data.iloc[:,:].values)
    fixData = [z for y in [[x]*x[2] for x in fixData] for z in y]
    completeData.append(fixData)

image = cv2.imread("bg-image.png")
out = cv2.VideoWriter("out3.avi",cv2.VideoWriter_fourcc(*'XVID'),20,(1440,900))

for i in range(1200):
    row = []
    #taking ith row of all users
    for j in range(20):
        row.append(completeData[j][i])
    #drawing heatmap
    fig = draw_heatmap(row, (1440,900), imagefile=image, durationweight=True, alpha=0.5)
    #converting mathplotlib figure to image and then video
    fig.canvas.draw( )

    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_rgb(), dtype=np.uint8 )
    buf.shape = ( w, h,3)

# canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    w, h, d = buf.shape
    mask = np.array(Image.frombytes( "RGB", ( w ,h ), buf.tostring( ) ))
    out.write(mask)
    print(i,j)
    plt.close("all")
