# -*- coding: UTF-8 -*-
"""
===============================================================
author：songdongdong
email：songdongdong@jd.com
date：2018
introduction: 用线性回归找到最佳拟合直线
===============================================================
"""

import numpy as np

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName,"r",encoding="utf-8").readline().split('\t')) - 1 #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr,yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:#计算行列式，防止为0 ，则不能求逆了
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    ws = np.linalg.solve(xTx,xMat.T * yMat)
    return ws
# 局部加权线性回归===========

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat
def rssError(yArr,yHatArr):
    return((yArr-yHatArr)**2).sum()

#岭回归，计算回归系数
def ridgeRegress(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx +np.eye(np.shape(xMat)[1])*lam
    if np.linalg.det(denom) == 0:#计算行列式，防止为0 ，则不能求逆了
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I *(xMat.T *yMat)
    return ws
def ridgeTest(xArr,yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    yMean = np.mean(yMat,0)
    yMat = yMat - yMean

    xMean = np.mean(xMat,0)
    xVar = np.var(xMat,0)
    xMat = (xMat -xMean)/xVar #归一化 所有的数据减去 均值再除以方差
    numTestPts = 30 #迭代多少次
    wMat = np.zeros((numTestPts,np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegress(xMat,yMat,np.exp(i-10))
        wMat[i,:] = ws.T
    return wMat
#归一化
def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = np.mean(inMat,0)   #calc mean then subtract it off
    inVar = np.var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat
def stageWis(xArr,yArr,eps=0.01,numIt=100):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    yMean = np.mean(yMat,0); yMat = yMat - yMean


    xMat = regularize(xMat)  # 归一化 所有的数据减去 均值再除以方差

    m,n = np.shape(xMat)
    returnMat = np.zeros((numIt,n))
    ws = np.zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowetError = np.inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE <lowetError:
                    lowetError = rssE
                    wsMax = wsTest
            ws = wsMax.copy()
            returnMat[i,:] = ws.T
    return returnMat


# -*-coding:utf-8 -*-
from bs4 import BeautifulSoup
def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
   """
   函数说明:从页面读取数据，生成retX和retY列表
   Parameters:
       retX - 数据X
       retY - 数据Y
       inFile - HTML文件
       yr - 年份
       numPce - 乐高部件数目
       origPrc - 原价
   Returns:
       无
   Website:
       http://www.cuijiahua.com/
   Modify:
       2017-12-03
   """
   # 打开并读取HTML文件
   with open(inFile, encoding='utf-8') as f:
       html = f.read()
   soup = BeautifulSoup(html)
   i = 1
   # 根据HTML页面结构进行解析
   currentRow = soup.find_all('table', r = "%d" % i)
   while(len(currentRow) != 0):
       currentRow = soup.find_all('table', r = "%d" % i)
       title = currentRow[0].find_all('a')[1].text
       lwrTitle = title.lower()
       # 查找是否有全新标签
       if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
           newFlag = 1.0
       else:
           newFlag = 0.0
       # 查找是否已经标志出售，我们只收集已出售的数据
       soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
       if len(soldUnicde) == 0:
           print("商品 #%d 没有出售" % i)
       else:
           # 解析页面获取当前价格
           soldPrice = currentRow[0].find_all('td')[4]
           priceStr = soldPrice.text
           priceStr = priceStr.replace('$','')
           priceStr = priceStr.replace(',','')
           if len(soldPrice) > 1:
               priceStr = priceStr.replace('Free shipping', '')
           sellingPrice = float(priceStr)
           # 去掉不完整的套装价格
           if  sellingPrice > origPrc * 0.5:
               print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
               retX.append([yr, numPce, newFlag, origPrc])
               retY.append(sellingPrice)
       i += 1
       currentRow = soup.find_all('table', r = "%d" % i)
#
def setDataCollect(retX, retY):
   """
   函数说明:依次读取六种乐高套装的数据，并生成数据矩阵
   Parameters:
       无
   Returns:
       无
   Website:
       http://www.cuijiahua.com/
   Modify:
       2017-12-03
   """
   scrapePage(retX, retY, 'setHtml/lego8288.html', 2006, 800, 49.99)                #2006年的乐高8288,部件数目800,原价49.99
   scrapePage(retX, retY, 'setHtml/lego10030.html', 2002, 3096, 269.99)                #2002年的乐高10030,部件数目3096,原价269.99
   scrapePage(retX, retY, 'setHtml/lego10179.html', 2007, 5195, 499.99)                #2007年的乐高10179,部件数目5195,原价499.99
   scrapePage(retX, retY, 'setHtml/lego10181.html', 2007, 3428, 199.99)                #2007年的乐高10181,部件数目3428,原价199.99
   scrapePage(retX, retY, 'setHtml/lego10189.html', 2008, 5922, 299.99)                #2008年的乐高10189,部件数目5922,原价299.99
   scrapePage(retX, retY, 'setHtml/lego10196.html', 2009, 3263, 249.99)                #2009年的乐高10196,部件数目3263,原价249.99

#交叉验证测试岭回归
def crossValidation(xArr,yArr,numVal = 10):
    m = len(yArr)
    indexList = range(m)
    indexList = [i for i in range(m)]
    errorMat = np.zeros((numVal,30))
    for i in range(numVal):
        trainX = [];trainY=[]
        testX = [];testY = []
        np.random.shuffle(indexList)
        #下面的循环是 按照 0.9的比例 放好 训练集合测试集
        for j in range(m):
            if j < m*0.9 :
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)#这里岭回归默认迭代30次，
        #
        for k in range(30):
            matTestX = np.mat(testX);
            matTrainX = np.mat(trainX)
            meanTrain = np.mean(matTrainX,0)
            varTrain = np.var(matTrainX,0)
            matTestX = (matTestX - meanTrain)/varTrain
            yEst = matTestX *np.mat(wMat[k,:]).T +np.mean(trainY)
            errorMat[i,k] = rssError(yEst.T.A,np.array(testY))
        meanErrors = np.mean(errorMat,0)
        minMean = float(min(meanErrors))
        bestWeights = wMat[np.nonzero(meanErrors==minMean)]
        xMat = np.mat(xArr);yMat = np.mat(yArr).T
        meanX = np.mean(xMat,0);varX = np.var(xMat,0)
        unReg = bestWeights/varX
        print("the best model from Ridge Regression is:\n", unReg)
        print("with constant term: ", -1 * sum(np.multiply(meanX, unReg)) + np.mean(yMat))




if __name__=="__main__":
   # xArr,yArr =  loadDataSet('ex0.txt')
   #线性回归 到最佳拟合直线=======================================
   # print(yArr)
   # print(xArr)
   # w = standRegres(xArr[0:2],yArr[0:2])
   # print(w)
   # xMat = np.mat(xArr)
   # yMat = np.mat(yArr)
   # yHat = xMat*w
   # # print("yHat",yHat)
   # import matplotlib.pyplot as plt
   #
   # fig = plt.figure()
   # ax = fig.add_subplot(111)
   # ax.scatter(xMat[:, 1].flatten().A[0],yMat.T[:,0].flatten().A[0])
   # # print(xMat[:, 1].flatten().A) #注意 flatten 和A的用法，
   # xCopy = xMat.copy()
   # # xCopy.sort(0)#防止出现混乱，绘图出现问题
   # yHat = xCopy*w
   # ax.plot(xCopy[:,1],yHat)
   # plt.show()
   # print(np.corrcoef( yHat.T,yMat)) #相关系数
   #局部加权线性回归=======================================
   # xArr, yArr = loadDataSet('ex0.txt')
   # print(yArr[0])
   # print(lwlr(xArr[0],xArr,yArr,1.0))
   # print(lwlr(xArr[0],xArr,yArr,0.001))
   # yHat = lwlrTest(xArr,xArr,yArr,0.01)
   # xMat = np.mat(xArr)
   # srtInd = xMat[:,1].argsort(0)#将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
   # xSort = xMat[srtInd][:,0,:]
   # print(xSort)
   # print(type(xSort),np.shape(xSort))
   # import matplotlib.pyplot as plt
   # fig = plt.figure()
   # ax = fig.add_subplot(111)
   # ax.plot(xSort[:,1],yHat[srtInd])
   # ax.scatter(xMat[:,1].flatten().A[0],np.mat(yArr).T.flatten().A[0],s=2,c='red')
   # plt.show()

   #预测鲍鱼的年龄
   # abX,abY =loadDataSet('abalone.txt')
   #
   # #训练集
   # print("训练集-")
   # yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
   # yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
   # yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
   #
   # print(rssError(abY[0:99],yHat01.T))
   # print(rssError(abY[0:99], yHat1.T))
   # print(rssError(abY[0:99], yHat10.T))
   #  #测试集  --效果
   # print("测试集-")
   # yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
   # yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
   # yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
   #
   # print(rssError(abY[100:199], yHat01.T))
   # print(rssError(abY[100:199], yHat1.T))
   # print(rssError(abY[100:199], yHat10.T))
   # print("简单数据集的比较")
   # ws = standRegres(abX[0:99],abY[0:99])
   # yHat = np.mat(abX[100:199]) *ws
   # print(rssError(abY[100:199],yHat.T.A))
   # print("================岭回归==============")
#    # abX,abY = loadDataSet('abalone.txt')
#    # ridgeWeights = ridgeTest(abX,abY)
#    # print(ridgeWeights)
#    # import matplotlib.pyplot as plt
#    # fig = plt.figure()
#    # ax = fig.add_subplot(111)
#    # ax.plot(ridgeWeights)
#    # plt.show()
#    print("================lasso==============")
#    xArr,yArr = loadDataSet('abalone.txt')
#    weMat = stageWis(xArr,yArr,0.001,5000)
#    print("和最小二乘法比较")
#    xMat = np.mat(xArr)
#    yMat = np.mat(yArr).T
#    xMat =regularize(xMat)
#    yM = np.mean(yMat,0)
#    yMat = yMat - yM
#    weights = standRegres(xMat,yMat.T)
#    print(weights.T)
#    import matplotlib.pyplot as plt
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.plot(weMat)
#    plt.show()
   print("预测乐高玩具套装的价格")
   lgX = [];lgY = []
   setDataCollect(lgX,lgY)
   m,n =np.shape(lgX)
   print(m,n)
   lgX1 = np.mat(np.ones((m,n+1)))#这里加一是因为，这里需要个常数项 1
   lgX1[:,1:5] = np.mat(lgX)
   print(lgX1[0],lgX[0])#检查数据是否准确
   ws = standRegres(lgX1,lgY)
   print(ws)
   print(lgX1[0]*ws)
   print("交叉验证")
   print(crossValidation(lgX,lgY,10))
   print(ridgeTest(lgX,lgY))








