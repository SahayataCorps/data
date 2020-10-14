
"""
This is a test file to check the original pyGazeAnalyser code
"""

import os
from pygazeanalyser.gazeplotter import draw_fixations, draw_heatmap, draw_scanpath, draw_raw
import numpy
import pandas as pd 

import matplotlib.pyplot as plt




def graphs(fixData):
    name = input("Enter name by which you want to save outputs: ")
    #draw_scanpath(sasFix, sas, (1440,900), imagefile="bg-image.png", alpha=0.5, savefilename=str(name)+"_scanOutNew.png")
    draw_fixations(fixData, (1440,900), imagefile="bg-image.png", durationsize=True, durationcolour=False, alpha=0.5, savefilename=str(name)+"_fixOutNew.png")
    draw_heatmap(fixData, (1440,900), imagefile="bg-image.png", durationweight=True, alpha=0.5, savefilename=str(name)+"_heatOutNew.png")

data1 = pd.read_csv("0_fixation_data.csv")
fixData1 = list(data1.iloc[:,:].values)
#draw_fixations(fixData1, (1366,685), imagefile="ProductAd.jpg", durationsize=True, durationcolour=False, alpha=0.5, savefilename="Product_fixOutNew.jpg")
draw_heatmap(fixData1, (1440,900), imagefile="bg-image.png", durationweight=True, alpha=0.5, savefilename="Product_heatOutNew.png")

# scanData = pd.read_csv("0_scanpath_data.csv")
# sas = list(scanData.iloc[:,1:].values)
# sasFix = list(scanData.iloc[:,1:6].values)
# draw_scanpath(sasFix, sas, (1440,900), imagefile="bg-image.png", alpha=0.5, savefilename="0_scanOutNew.png")

# sasData = []
# fixData = []
# for i in range(20):
#     scanData = pd.read_csv("200CSV/"+str(i)+"_scanpath_data.csv")
#     sas = list(scanData.iloc[:,1:].values)
#     sasFix = list(scanData.iloc[:,1:6].values)
#     for x in range(len(sas)):
#         sasData.append(sas[x])
#         fixData.append(sasFix[x])
# draw_fixations(fixData, (1440,900), imagefile="bg-image.png", durationsize=True, durationcolour=False, alpha=0.5, savefilename="MultiUserFixation.jpg")
# draw_scanpath(fixData, sasData, (1440,900), imagefile="bg-image.png", alpha=0.5, savefilename="complete_scanOutNew.png")


#data2 = pd.read_csv("1_fixation_data.csv")
#fixData2 = list(data2.iloc[:,:].values)
# scanData2 = pd.read_csv("1_scanpath_data.csv")
# sas2 = list(scanData.iloc[:,:].values)
# sasFix2 = list(scanData.iloc[:,0:5].values)


# avgFix = list((fixData1 + fixData2)/2)
# graphs(avgFix)
#draw_fixations(fixData1, (1440,900), imagefile="bg-image.png", durationsize=True, durationcolour=False, alpha=0.5, savefilename="_fixOutNew.png")
#draw_fixations(fixData1, (1440,900), imagefile="bg-image.png", durationweight=True, alpha=0.5, savefilename="avg_heatOutNew.png")