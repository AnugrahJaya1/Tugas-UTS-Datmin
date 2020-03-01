# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:35:26 2020

@author: anugrahjaya1
"""
# untuk baca gambar
import cv2
import math
import numpy
from sklearn.cluster import KMeans
import pandas as pd
# define data
import _pickle as cPickle


def convertImagetoArrayBlock16(fullFileName):
    clt = KMeans(n_clusters = 3)
    img = cv2.imread(fullFileName)
        
    #convert ke cielab
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        
    # membagi gambar menajdi 16 blok
    size = img.shape
    width = math.floor(size[1]/4)
    height = math.floor(size[0]/4)
           
    idx = 0
        
    arrayGambar = [];
    # masuk ke baris pertama
    for i in range(0,4):
        #masuk ke kolom pertama
        for j in range(0,4):
            cropped_img = lab[j*height:height*(j+1), i*width:width*(i+1)]
            clt.fit(cropped_img[0])
                
            #centroid 
            centroid = clt.cluster_centers_
            
            # get all the labels
            arr  =  clt.labels_ 
            count = numpy.bincount(arr) 
            idx = idx+1
            maxCentroid = 0
            for row in count:
                if count[maxCentroid] < row:
                    maxCentroid = maxCentroid + 1
         
            arrayGambar.append(centroid[maxCentroid])
            
    return arrayGambar




# read the dataset 
arrayUtamaDataset = cPickle.load( open( "arrayOfImage.pkl", "rb" )) 


# read Image and convert into 16block of LAB 
inputImageFileName = convertImagetoArrayBlock16('inputImage.jpg')

hasilKNN = pd.DataFrame({'jarak':[], 'label':[]})
# untuk pergambar
for i in range(0,451):
    block = [];
    
    # untuk satu block
    for j in range (0,16):
        jarak = 0;
        for k in range (0,3):
            jarak = jarak + math.pow(arrayUtamaDataset[i][j][k]-inputImageFileName[j][k],2)
            
        # nilai satu block
        jarak = math.sqrt(jarak)
        block.append(jarak)
        
    
    # rata rata satu gambar
    avg = numpy.mean(block)
    namaLabel = arrayUtamaDataset[i][16]
    hasilKNN = hasilKNN.append({'jarak':avg,'label':namaLabel}, ignore_index=True)
    

# sort by jarak
hasilKNN = hasilKNN.sort_values(by='jarak')

k = math.floor(hasilKNN.shape[0]/3)

hasilKNN = hasilKNN.iloc[0:k]

countLabel = hasilKNN.groupby('label').count()

prediksi = countLabel.index[0]

print('Gambar inputImage.jpg termasuk : '+prediksi)