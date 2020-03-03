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
import os
# define data
import _pickle as cPickle



# method untuk menghitung euclidian distance
def euclidianDistance(dataset, inputImg):
    
    temp = []
    for x in range(0,16):
        jarak = 0 
        
        for k in range (0,3):
            jarak = jarak + math.pow(dataset[j][k]-inputImg[j][k],2)
                
            # nilai satu block
        jarak = math.sqrt(jarak)
        temp.append(jarak)
        
    return temp


# method untuk melakukan perhitungan cosine simalarity
def cosineSimilarity(dataset, inputImg):
    temp = []
    for x in range(0,16):
        t = 0 
        a = 0
        b = 0 
        
        for k in range (0,3):
            t = t + dataset[j][k]*inputImg[x][k]
            a = a + math.pow(dataset[j][k],2)
            b = b + math.pow(inputImg[x][k],2)
            
        temp.append(t/(math.sqrt(a)*math.sqrt(b)))
            
    return temp

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

    
imageClassFolderName = os.getcwd()+'\\'+'test img'
    
    
print(imageClassFolderName)  
arrOfImage = os.listdir(imageClassFolderName)
for filename in arrOfImage:
    print(filename)
    

    # read Image and convert into 16block of LAB 
    inputImageFileName = convertImagetoArrayBlock16(imageClassFolderName+'\\'+filename)
    
    # dataframe penampung
    hasilKNN = pd.DataFrame({'score':[], 'label':[]})
    
    # untuk pergambar
    for i in range(0,451):
        block = [];
        
        # untuk satu block
        for j in range (0,16):
            #jika menggnakan euclidian distance
            #jarak = euclidianDistance(arrayUtamaDataset[i], inputImageFileName)
            
            #jika menggunakan 1 / euclidian distance
            jarak = euclidianDistance(arrayUtamaDataset[i], inputImageFileName)
            
            #jika menggunakan cosine similarity
            #jarak = cosineSimilarity(arrayUtamaDataset[i], inputImageFileName)
            jarak = numpy.mean(jarak)
            
            # menambahkan hasil perhitungan jarak untuk tiap blok
            #jika menggunakan 1 / euclidian distance
            block.append(1/jarak)
            #cosine 
            #block.append(jarak)
            
        
        # rata rata satu gambar
        score = numpy.sum(block)
        
        # untuk mendapatkan nama label gambar
        namaLabel = arrayUtamaDataset[i][16]
        
        # memasukkan hasil perhitungan kedalam dataframe 
        hasilKNN = hasilKNN.append({'score':score,'label':namaLabel}, ignore_index=True)
      
    
    # sort by jarak
    
    # untuk euclidian distance (BUKAN UNTUK 1 / euclidian distance)
    #hasilKNN = hasilKNN.sort_values(by='jarak')
    
    # untuk cosine similarity dan 1 / euclidiance
    hasilKNN = hasilKNN.sort_values(by='score', ascending=False)
    
    # pembulatan kebawah, untuk mendapatkan 1/3 jumlah data
    k = math.floor(hasilKNN.shape[0]/3)
    
    # mengambil 1/3 dari data
    hasilKNN = hasilKNN.iloc[0:k]
    
    # menghitung jumlah label dari 1/3 data
    countLabel = hasilKNN.groupby('label').sum().sort_values(by='score',ascending=False)
    
    # mengambil label dengan jumlah anggota terbanyak
    prediksi = countLabel.index[0]
    
    # print hasil
    print('Gambar '+filename+' termasuk : '+prediksi)