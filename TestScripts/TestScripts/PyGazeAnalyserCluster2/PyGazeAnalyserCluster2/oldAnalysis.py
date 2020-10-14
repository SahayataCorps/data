import os
from pygazeanalyser.gazeplotter import draw_fixations, draw_heatmap, draw_scanpath, draw_raw
import numpy
import pandas as pd 
data = pd.read_csv("data.csv")
#edfdataX = data.iloc[:,0].values
#edfdataY = data.iloc[:,1].values
fixData = data.iloc[:1200,:].values
def fix(fData):
	point = []
	finalData = []
	durr = 0
	st=0
	for p in fData:
		st+=1
		if len(point)==0:
			point.append(p[0])
			point.append(p[1])
			durr+=1
		else:
			if p[0] ==point[0] and p[1]==point[1]:
				durr+=1
			else:
				finalData.append([st-durr,st,durr, point[0], point[1]])
				point = []
				durr=0
	
	return finalData
def sas(fData):
	point = []
	finalData = []
	durr = 0
	st=0
	for p in fData:
		st+=1
		if len(point)==0:
			point.append(p[0])
			point.append(p[1])
			durr+=1
		else:
			if p[0] ==point[0] and p[1]==point[1]:
				durr+=1
			else:
				if p[0]==0 and p[1]==0:
					p[0] = point[0]
					p[1] = point[1]
				finalData.append([st-durr,st,durr, point[0], point[1],p[0],p[1]])
				point = []
				durr=0
	return finalData

#draw_raw(edfdataX, edfdataY, (1440, 900), imagefile="bg-image.png", savefilename="fdOutput.png")
fixtData = fix(fixData)
sasData = sas(fixData)
draw_scanpath(fixtData, sasData, (1440,900), imagefile="bg-image.png", alpha=0.5, savefilename="scanOut.png")
draw_fixations(fixtData, (1440,900), imagefile="bg-image.png", durationsize=True, durationcolour=False, alpha=0.5, savefilename="fixOut.png")
draw_heatmap(fixtData, (1440,900), imagefile="bg-image.png", durationweight=True, alpha=0.5, savefilename="heatOut.png")






"""
We can show all users fixation points on the image
We can show all users heatmaps on the image
For sasscade: We will show the top n most followed paths.
Others: Most common starting point, most common ending point, 

  
"""