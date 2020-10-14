
"""This is a test file to create scanpath for single user on video."""
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

#Read data
scanData = pd.read_csv("0_scanpath_data.csv")
sas = list(scanData.iloc[:,1:].values)
sasFix = list(scanData.iloc[:,1:6].values)

#Read poll image
image = cv2.imread("bg-image.png")
out = cv2.VideoWriter("out3.avi",cv2.VideoWriter_fourcc(*'XVID'),20,(1440,900))

i=0
x = 0
etime  = 200
dataSas = []
dataFix = []
while i<200:
    i+=1
    rowSas = sas[x] # row scanpath/ sascade
    rowFix = sasFix[x] #row scanpath-fixation
    etime = rowSas[1] #end time
    fig = draw_scanpath([rowFix], [rowSas], (1440,900), imagefile=image, alpha=0.5) #draw scanpath on the frame
    # convert matplotlib figure to images
    fig.canvas.draw( ) 
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_rgb(), dtype=np.uint8 )
    buf.shape = ( w, h,3)
    buf = np.roll ( buf, 3, axis = 2 )
    w, h, d = buf.shape
    mask = np.array(Image.frombytes( "RGB", ( w ,h ), buf.tostring( ) ))
    out.write(mask)
    print(i)
    plt.close("all")
    if i==etime:
        x+=1