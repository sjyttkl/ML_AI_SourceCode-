# -*- coding: UTF-8 -*-
"""
===============================================================
author：songdongdong
email：songdongdong@jd.com
date：2018
introduction:svm
===============================================================
"""
import numpy as np

def loadDataSet(fileName):
    dataMat=  []
    labelMat = []
    fr = open(fileName,"r",encoding="utf-8")
    for line in fr.readlines():
        lineArr = line.strip().split("\t")

        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j = i
    while(j==i):
        j = int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj


def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    datamatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()



if __name__=="__main__":
    dataArr,labelArr = loadDataSet('testSet.txt')
    print(labelArr)