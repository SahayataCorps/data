"""
This is a test file to create graphs for images. Used in creating final script for deployment.
"""



from gazePlotterNew import *
import pandas as pd
import numpy as np
import cv2
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans


#testing single user on image

def filteredData(name, filters):
    pass
def drawSUImage(name, imageFile, isFixation = True, isHeatMap = True, isScanpath = True  ):
    data = pd.read_csv(name+"_fixation_data.csv")
    fixData = list(data.iloc[:,:].values)
    image = cv2.imread(imageFile)
    dispSize = [image.shape[1], image.shape[0]]
    fix = singleParseFixations(fixData)
    if isFixation:
        plotFixation(fix, dispSize, imageFile, alpha=0.5, savefilename=imageFile.split(".")[0]+"_"+name+"_fixation_single.jpg") 
        print("Fixation Done")
    if isHeatMap:
        plotHeatMap(fix,dispSize, imageFile ,durationweight=True, alpha=0.5, savefilename=imageFile.split(".")[0]+"_"+name+"_heatmap_single.jpg")
        print("Heatmap Done")
    if isScanpath:
        scanData = pd.read_csv(name+"_scanpath_data.csv")
        print(scanData.head(5))
        sas = list(scanData.iloc[:,1:].values) #index in csv
        sasFix = list(scanData.iloc[:,1:6].values) #index in csv
        sasFix2 = singleParseFixations(sasFix)
        plotScanpath(sasFix2, sas, dispSize, imageFile, alpha=0.5, savefilename=imageFile.split(".")[0]+"_"+name+"_scanpath_single.jpg", withFixation = True, withHeatmap = False, singleUser= True)
        plotScanpath(sasFix2, sas, dispSize, imageFile, alpha=0.5, savefilename=imageFile.split(".")[0]+"_"+name+"_scanpath_heatmap_single.jpg", withFixation = False, withHeatmap = True, singleUser= True)
        print("Scanpath Done")

def drawMUImage(name, filters, imageFile, isFixation = True, isHeatMap = True, isScanpath = True ):
    data = filteredData(name, filters)
    fixData = list(data.iloc[:, :].values)
    image = cv2.imread(imageFile)
    dispSize = [image.shape[1], image.shape[0]]
    fix = singleParseFixations(fixData)
    if isFixation:
        plotFixation(fix, dispSize, imageFile, alpha=0.5, savefilename=imageFile.split(".")[0] + "_" + name + "_fixation_single.jpg")
        print("Fixation Done")
    if isHeatMap:
        plotHeatMap(fix, dispSize, imageFile, durationweight=True, alpha=0.5, savefilename=imageFile.split(".")[0] + "_" + name + "_heatmap_single.jpg")
        print("Heatmap Done")
    if isScanpath:
        k = 8
        completeDataScan = np.array(data)
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=10, random_state=0)
        pred_y = kmeans.fit_predict(completeDataScan[:, 4:])
        centers = [(round(i, 0), round(j, 0)) for (i, j) in [list(x) for x in kmeans.cluster_centers_]]
        clusters = {}
        for i in range(len(pred_y)):
            cen = centers[pred_y[i]]
            point = completeDataScan[i]
            cenData = clusters.get(cen, [0, 0, 0])
            nop = cenData[2] + 1
            cost = (cenData[0] * (nop - 1) + point[0] + 1) // nop
            durr = (cenData[1] * (nop - 1) + point[3]) // nop
            cenData = [cost, durr, nop]
            clusters[cen] = cenData

        scanData = []
        for (x, y) in clusters:
            pointData = [x, y] + clusters[(x, y)]
            scanData.append(pointData)

        scanData.sort(key=lambda x: x[-1], reverse=True)
        scanData.sort(key=lambda x: x[2])

        fixData = list(map(list, scanData))
        for i in range(len(scanData)):
            if i == len(scanData) - 1:
                scanData[i] += [scanData[i][0], scanData[i][1]]
            else:
                scanData[i] += [scanData[i + 1][0], scanData[i + 1][1]]



drawSUImage("0","bg-image.png")