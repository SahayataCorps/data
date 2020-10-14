"""This is a test file to create Scanpath of multiusers on image using K-mean clustering."""
from pygazeanalyser.gazeplotterMUI import draw_fixations, draw_heatmap, draw_scanpath, draw_raw, multiuser_image_scanpath_heatmap 
import numpy as np
import pandas as pd 
import cv2
import matplotlib.pyplot as plt

#read data
dataScan = []
for x in range(20):
    data = pd.read_csv("200CSV/"+str(x)+"_scanpath_data.csv")
    scData = list(data.iloc[:,:6].values)
    scData = [list(x) for x in scData]
    dataScan+=scData

#dimension of poll image
width = 1440
height = 900
k = 8
completeDataScan = np.array(dataScan)

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

#creating clusters using K-mean
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(completeDataScan[:,4:])

#finding centers of clusters form by K-mean
centers = [(round(i,0),round(j,0)) for (i,j) in[list(x) for x in kmeans.cluster_centers_]]
clusters = {}

#finding cost of the clusters
for i in range(len(pred_y)):
    cen = centers[pred_y[i]]
    point = completeDataScan[i]
    cenData = clusters.get(cen,[0,0,0])
    nop = cenData[2]+1
    cost = (cenData[0]*(nop-1) + point[0]+1 )//nop
    durr = (cenData[1]*(nop-1) + point[3])//nop
    cenData = [cost,durr,nop]
    clusters[cen] = cenData

#aligning clusters
scanData = []
for (x,y) in clusters:
    pointData = [x,y]+clusters[(x,y)]
    scanData.append(pointData)

scanData.sort(key = lambda x: x[-1], reverse=True)
scanData.sort(key = lambda x: x[2])

fixData = list(map(list, scanData))
for i in range(len(scanData)):
    if i==len(scanData)-1:
        scanData[i]+=[scanData[i][0],scanData[i][1]]
    else:
        scanData[i]+=[scanData[i+1][0],scanData[i+1][1]]
print(scanData)

#drawing scanpath with heatmap
multiuser_image_scanpath_heatmap(dataScan, fixData, scanData, (1440,900), imagefile="bg-image.png", alpha=0.5, savefilename="MultipleUserImageScanpathHeatmap8.png")
