# -*- coding: UTF-8 -*-
"""
===============================================================
author：songdongdong
email：songdongdong@jd.com
date：2018.09.23
introduction:pca降纬
===============================================================
"""
__author__ = "songdongdong"


import numpy as np
def loadDataSet(fileName,delim="\t"):
    fr= open(fileName)
    strigArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr =[list(map(float,line)) for line in strigArr]
    # datArr = list(datArr)
    return np.mat(datArr)

def pca(datamat,topNfeat=9999999):
    meanVals=np.mean(datamat,axis=0)
    meanRemoved = datamat-meanVals#去平均值
    covMat=np.cov(meanRemoved,rowvar=False) #就算协方差矩阵
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))#计算特征向量和特征值的方法
    eigValInd = np.argsort(eigVals)
    # 从小到大对N个值排序
    eigValInd = eigValInd[:-(topNfeat+1):-1]#只取前多少个
    redEigVects = eigVects[:,eigValInd]
    # 将数据转换到新空间
    lowDDataMat = meanRemoved * redEigVects#将数据转换到新的空间
    reconMat = (lowDDataMat*redEigVects.T)+meanVals#这里是还原
    return lowDDataMat,reconMat
#半导体制作降纬
def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = np.shape(datMat)[1]
    for i in range(numFeat):
        print(np.isnan(datMat[:, i]))
        print(np.isnan(datMat[:, i].A))
        print(np.nonzero(~np.isnan(datMat[:, i])))  # A表示的意思是矩阵转为数组array
        print(datMat[np.nonzero(~np.isnan(datMat[:, i].A))[0], i])
        meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[np.nonzero(np.isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat
if __name__=="__main__":
    datamat = loadDataSet("testSet3.txt")
    lowDMat,reconMat = pca(datamat,1)
    print(np.shape(lowDMat))
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datamat[:,0].flatten().A[0],datamat[:,1].flatten().A[0],marker='^',s=90)
    ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c="red")
    plt.show()
   #半导体制作
    #均值
    datamat2 = replaceNanWithMean()
    meanVals = np.mean(datamat2,axis=0)
    meanRemoved = datamat2-meanVals
    #计算协方差
    covMat = np.cov(meanRemoved,rowvar=False)
    #进行特征分析和分解
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))
    print(eigVals,eigVects)





















