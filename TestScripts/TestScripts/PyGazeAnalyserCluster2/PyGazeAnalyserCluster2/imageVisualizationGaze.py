"""
This is the main file that was deployed on frontend.
This was used to create all the graphs for poll Image.
Code for filter was remaining in the multi-users case
"""

from gazePlotterNew import *
import pandas as pd
import numpy as np
import cv2
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import argparse

def filteredData(name, filters):
    pass

#To draw graph for single user image
def drawSUImage(name, imageFile, isFixation = True, isHeatMap = True, isScanpath = True  ):
    #read data
    data = pd.read_csv("controllers/"+name+"_fixation_data.csv")
    fixData = list(data.iloc[:,:].values)
    #read image
    image = cv2.imread("controllers/"+imageFile)
    #size of image
    dispSize = [image.shape[1], image.shape[0]]
    #function in gazePlotterNew to convert data into dictionary format and remove unwanted data
    fix = singleParseFixations(fixData)

    if isFixation:
        plotFixation(fix, dispSize, "controllers/"+imageFile, alpha=0.5, savefilename="public/"+name+"_fixation.jpg")
        print("Fixation Done")
    if isHeatMap:
        plotHeatMap(fix,dispSize, "controllers/"+imageFile ,durationweight=True, alpha=0.5, savefilename="public/"+name+"_heatmap.jpg")
        print("Heatmap Done")
    if isScanpath:
        scanData = pd.read_csv("controllers/"+name+"_scanpath_data.csv")
        print(scanData.head(5))
        sas = list(scanData.iloc[:,1:].values) #index in csv
        sasFix = list(scanData.iloc[:,1:6].values) #index in csv
        sasFix2 = singleParseFixations(sasFix)
        plotScanpath(sasFix2, sas, dispSize, "controllers/"+imageFile, alpha=0.5, savefilename="public/"+name+"_scanpath.jpg", withFixation = True, withHeatmap = False, singleUser= True)
        plotScanpath(sasFix2, sas, dispSize, "controllers/"+imageFile, alpha=0.5, savefilename="public/"+name+"_heatscan.jpg", withFixation = False, withHeatmap = True, singleUser= True)
        print("Scanpath Done")

#To draw graph for multiple users image
def drawMUImage(name, filters, imageFile, isFixation = True, isHeatMap = True, isScanpath = True ):
    #read data
    data = filteredData(name, filters)
    fixData = list(data.iloc[:, :].values)
    #read image
    image = cv2.imread(imageFile)
    #image size
    dispSize = [image.shape[1], image.shape[0]]
    #function in gazePlotterNew to convert data into dictionary format and remove unwanted data
    fix = singleParseFixations(fixData)
    if isFixation:
        plotFixation(fix, dispSize, imageFile, alpha=0.5, savefilename=imageFile.split(".")[0] + "_" + name + "_fixation_single.jpg")
        print("Fixation Done")
    if isHeatMap:
        plotHeatMap(fix, dispSize, imageFile, durationweight=True, alpha=0.5, savefilename=imageFile.split(".")[0] + "_" + name + "_heatmap_single.jpg")
        print("Heatmap Done")
    if isScanpath:
        k = 8
        #cluster formations
        completeDataScan = np.array(data)
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=10, random_state=0)
        pred_y = kmeans.fit_predict(completeDataScan[:, 4:])
        centers = [(round(i, 0), round(j, 0)) for (i, j) in [list(x) for x in kmeans.cluster_centers_]]
        clusters = {}

        #Cost calculation
        for i in range(len(pred_y)):
            cen = centers[pred_y[i]]
            point = completeDataScan[i]
            cenData = clusters.get(cen, [0, 0, 0])
            nop = cenData[2] + 1
            cost = (cenData[0] * (nop - 1) + point[0] + 1) // nop
            durr = (cenData[1] * (nop - 1) + point[3]) // nop
            cenData = [cost, durr, nop]
            clusters[cen] = cenData
        
        #Re-aligning custers according to cost
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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-client', type=str, help='client_id')
    parser.add_argument('-content', type=str, help='content_id')
    parser.add_argument('-age', type=str, help='age')
    parser.add_argument('-gender', type=str, help='gender')
    parser.add_argument('-user', type=str, help='user_id')
    parser.add_argument('-image', type=str, help='image name')
    args = parser.parse_args()

    if  args.user is not None:
        fileName = args.client+"_"+args.content+"_All_All_"+args.user
        drawSUImage(fileName,args.image)
