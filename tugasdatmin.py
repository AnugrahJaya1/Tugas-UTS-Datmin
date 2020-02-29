# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:50:04 2020

@author: anugrahjaya1
"""

# untuk baca gambar
import cv2
import math
import numpy
from sklearn.cluster import KMeans
import os 
from numpy import asarray
from numpy import save
# define data
import _pickle as cPickle

# Method to convert image into multidimensional array that contains 16 block with representative 
# parameter : string file gambar 
# return : arraynumpy 
def convertImagetoArrayBlock16(fullFileName,imageClassFolderName):
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
        
    
    arrayGambar.append(imageClassFolderName)        
    return arrayGambar
 



arrayUtama = []
# list of all images class name
listOfImageClass = os.listdir(os.getcwd()) 
for classDir in listOfImageClass:
    imageClassFolderName = os.getcwd()+ "\\" + classDir
    
    
    arrOfImage = os.listdir(imageClassFolderName)
    for filename in arrOfImage:
        fullFileName = imageClassFolderName+ '\\' + filename
        arrayGambar = convertImagetoArrayBlock16(fullFileName,classDir)
        arrayUtama.append(arrayGambar)


# from the all the image in the dataset that already been converted
# to array of image and array of image consist of array of block that contain 
# color representatives 


# write to cPickle
cPickle.dump( arrayUtama, open( "arrayOfImage.pkl", "wb" ) )

