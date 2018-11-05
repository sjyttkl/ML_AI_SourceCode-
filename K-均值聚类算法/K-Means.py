# -*- coding: UTF-8 -*-
"""
===============================================================
author：songdongdong
email：695492835@qq.com
date：2018.09.16
introduction:
===============================================================
"""
__author__ = "songdongdong"
import numpy as np
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # 将每行映射成浮点数，python3返回值改变，所以需要
        dataMat.append(fltLine)
    return dataMat

#计算欧式距离
def distEclud(vecA,vecB):
    return np.sqrt(np.sum(np.power(vecA-vecB,2)))
#为给定的数据集构建一个包含k个随机质心的集合
def randCent(dataSet,k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangJ = float(np.max(dataSet[:,j])-minJ)
        centroids[:,j] = minJ +rangJ * np.random.rand(k,1)#确保随机点在数据的边界之内
        # print(minJ+rangJ,np.random.rand(k,1),centroids[:,j])
    return centroids
#k-均值聚类算法,
def kMeans(dataSet,k,disMeas=distEclud,createCent=randCent ):
    m = np.shape(dataSet)[0] #行
    clusterAssment=np.mat(np.zeros((m,2)))#用来存储每个点的簇分配结果，第一列是索引值，第二列的是误差值。这里误差是值当前点到簇质心的距离，后面会使用这个误差值来评价聚类效果
    centroids= createCent(dataSet,k)
    clusterChanged=True
    #遍历所有数据找到距离每个点最近的质心，t通过计算点到质心的距离来完成
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf;minIndex = -1
            #寻找最近的质心
            for j in range(k):
                distJI = disMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI;minIndex = j
            if clusterAssment[i,0] !=minIndex:clusterChanged = True#直到所有簇分配不再改变为止
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        #更新质心位置
        for cent in  range(k):
            # print(np.nonzero(clusterAssment[:, 0].A == cent))这里得到的是位置信息，前面是行，后面是列，结合在一起就是点。
            # print("===========")
            # print(np.nonzero(clusterAssment[:,0].A==cent)[0])#这里是获得行，
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]] #A代表将矩阵转化为array数组类型
            centroids[cent,:] = np.mean(ptsInClust,axis=0)
    return centroids,clusterAssment

#k-均值聚类算法虽然可以收敛，但是是局部最小，不是全局最小。用于度量全局最小的是SSE,，误差平方和
#SSE值越小表示数据点越接近质心聚类效果越好，因此对误差取了平方，因此更加原离那些远距离中心的点。
#一种肯定   可以降低SSE值的办法是增加簇的个数，但是这违背了聚类的目标。聚类的目标是在簇不变的情况下提高簇的质量。
#现在是对簇进行后处理。
#一种方法是：将具有最大SSE值的簇划分为两个簇，。具体实现可以将最大簇包含的点过滤出并且在这些点上运行k-均值聚类算法。k设置为2
#第二种办法是为了簇总数保持不变，可以将两个簇合并：
   #a,合并最近的质心。  思路是通过计算所有质心之间的距离，然后合并距离最近的两个点来实现。
   #b,需要合并两个簇，然后计算总的SSE值。
#必须在所有的簇上面重复上述过程，直到找到最佳簇为止

#二分k-均值聚类算法
def bikMeans(dataSet,k,disMeas=distEclud):
    m = np.shape(dataSet)[0]
    clusterAssement = np.mat(np.zeros((m,2)))
    # 创建一个初始簇
    centroid0 = np.mean(dataSet,axis=0).tolist()[0]
    centList = [centroid0]
    for j in  range(m):
        clusterAssement[j,1] = disMeas(np.mat(centroid0),dataSet[j,:])**2
    while(len(centList)<k):
        lowestSSE = np.inf
        #尝试划分每一个簇
        for i in range(len(centList)):
            # print(np.nonzero(clusterAssement[:,0].A==i))
            ptsInCurrCluster =  dataSet[np.nonzero(clusterAssement[:,0].A==i)[0],0]
            centroidMat,splitClustAss =kMeans(ptsInCurrCluster, 2,disMeas)
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit = sum(clusterAssement[np.nonzero(clusterAssement[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)#将未划分的误差和划分的误差之和作为本次划分的误差
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        #更新簇的分配结果
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # change 1 to 3,4, or whatever
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))

        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # replace a centroid with two best centroids
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssement[np.nonzero(clusterAssement[:, 0].A == bestCentToSplit)[0],:] = bestClustAss  # reassign new clusters, and SSE
    return np.mat(centList), clusterAssement
#===yahoo==============
import  urllib.request
import urllib.parse


import json


def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  # create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'  # JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.parse.urlencode(params)
    yahooApi = apiStem + url_params  # print url_params
    print(yahooApi)
    c = urllib.request.urlopen(yahooApi)
    return json.loads(c.read())


from time import sleep


def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:
            print("error fetching")
        sleep(1)
    fw.close()


def distSLC(vecA, vecB):  # Spherical Law of Cosines
    a = np.sin(vecA[0, 1] * np.pi / 180) * np.sin(vecB[0, 1] * np.pi / 180)
    b = np.cos(vecA[0, 1] * np.pi / 180) * np.cos(vecB[0, 1] * np.pi / 180) * \
        np.cos(np.pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return np.arccos(a + b) * 6371.0  # pi is imported with numpy


import matplotlib
import matplotlib.pyplot as plt


def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(datList)
    myCentroids, clustAssing = bikMeans(datMat, numClust, disMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', \
                      'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()


if __name__ == "__main__":
    datMat = np.mat(loadDataSet("testSet.txt"))
    #======测试使用========
    # print(np.shape(datMat))
    # print(min(datMat[:,0]))
    # print(min(datMat[:,1]))
    # print(max(datMat[:,0]))
    # print(max(datMat[:,1]))
    # print(randCent(datMat,2))
    # print(distEclud(datMat[0],datMat[1]))
    #=====k-均值聚类算法
    # datMat = np.mat(loadDataSet("testSet.txt"))
    # myCentroids,clustAssing = kMeans(datMat,4)
    # print(myCentroids,clustAssing)
    # =====二分-k-均值算法
    datMat3 =   np.mat(loadDataSet('testSet2.txt'))
    # print(datMat3)
    centList,myNewAssments = bikMeans(datMat3,3)
    print(centList)
    #===Yahoo==
    # goResults = geoGrab("1  VA Center",'Augusta,ME')#不能运行
    #直接使用places.txt文件进行聚类吧。
    clusterClubs(5)
