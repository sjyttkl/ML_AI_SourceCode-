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

"""
    读取数据
    Parameters:
        fileName - 文件名
    Returns:
        dataMat - 数据矩阵
        labelMat - 数据标签
    """
def loadDataSet(fileName):
    dataMat=  []
    labelMat = []
    fr = open(fileName,"r",encoding="utf-8")
    for line in fr.readlines():
        lineArr = line.strip().split("\t")

        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

"""
    函数说明:随机选择alpha_j的索引值

    Parameters:
        i - alpha的下标
        m - alpha参数个数
    Returns:
        j - alpha_j的索引值
    """
def selectJrand(i,m):
    j = i#we want to select any J not equal to i
    while(j==i):
        j = int(np.random.uniform(0,m))
    return j
"""
    调整大于H或小于L的alpha值
    Parameters：
        oS - 数据结构
        k - 标号为k的数据的索引值
    Returns:
        aj - 修剪后的alpah_j的值
    """
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj



'''
简化版的SMO函数
'''
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):  # 输入数据，标记，常数C，容错率，最大迭代次数
    dataMatrix =np.mat(dataMatIn);   # 转换成矩阵
    labelMat = np.mat(classLabels).transpose()  # 转换成矩阵，并转置，标记成为一个列向量，每一行和数据矩阵对应
    m,n = np.shape(dataMatrix)  # 行，列

    b = 0  # 参数b的初始化
    alphas = np.mat(np.zeros((m,1)))  # 参数alphas是个list，初始化也是全0，大小等于样本数
    iter = 0  # 当前迭代次数，maxIter是最大迭代次数

    while (iter < maxIter):  # 当超过最大迭代次数，推出
        alphaPairsChanged = 0  # 标记位，记录alpha在该次循环中，有没有优化
        for i in range(m):  # 第i个样本
            fXi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b  # 第i样本的预测类别
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions  # 误差

            #是否可以继续优化
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)  # 随机选择第j个样本
                fXj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b  # 样本j的预测类别
                Ej = fXj - float(labelMat[j])  # 误差

                alphaIold = alphas[i].copy()  # 拷贝，分配新的内存
                alphaJold = alphas[j].copy()

                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L==H: print ("L==H"); continue

                # 这个是二阶导数求解
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T

                if eta >= 0: print ("eta>=0"); continue

                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)  # 门限函数阻止alpha_j的修改量过大

                #如果修改量很微小
                if (abs(alphas[j] - alphaJold) < 0.00001): print ("alpha_j变化太小了。。j not moving enough"); continue

                # alpha_i的修改方向相反
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                # 为两个alpha设置常数项b
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0

                # 说明alpha已经发生改变
                alphaPairsChanged += 1
                print ("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))

        #如果没有更新，那么继续迭代；如果有更新，那么迭代次数归0，继续优化
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print ("iteration number: %d" % iter)

    # 只有当某次优化更新达到了最大迭代次数，这个时候才返回优化之后的alpha和b
    return b,alphas

class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2))) #误差缓存,第一列是是否有效的标记位，第二位是实际的值

    #计算误差
def calcEk(oS,k):
    fXk= float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:])) +oS.b
    Ek = fXk-float(oS.labelMat[k])
    return Ek

def selectJ(i,oS,Ei):   #内循环中的启发式方法
    maxK=-1;maxDeltaE = 0;Ej = 0;
    oS.eCache[i] = [1,Ei]
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    if(len(validEcacheList))>1:
        for k in validEcacheList:
            if k==i :continue
            Ek = calcEk(oS,k)
            deltaE = np.abs(Ei-Ek)
            if (deltaE >maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK,Ej
    else:
        j = selectJrand(i,oS.m)
        Ej= oS.calcEk(oS,j)
    return j,Ej
def updateEk(oS,k):
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]

def innerL(i,oS):
    Ei = calcEk(oS,i)
    if((oS.labelMat[i]*Ei <-oS.tol) and (oS.alphas[i]<oS.c) or ((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>0))):
        j,Ej = selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold =oS.alphas[j].copy()
        if(oS.labelMat[i]!=oS.labelMat[j]):
            L= np.max(0,oS.alphas[j] - oS.alphas[i])
            H = np.min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L=np.max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H = np.min(oS.C,oS.alphas[j]+oS.alphas[i])
        if L==H :print("L == H") ;return 0
        eta = 2.0*oS.X[j,:]*oS.X[i,:].T -oS.X[i,:]*oS.X[i,:].T-oS.X[j,:]*oS.X[j,:].T
        if eta>=0:print ("eta>0");return 0;
        oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j) #更新误差缓存
        if(np.abs(oS.alphas[j]-alphaJold) < 0.00001):
            print("j no moving enough "); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        updateEk(oS,i) #更新误差缓存

        b1 = oS.b -Ei-oS.labelMat[i]*(oS.alphas[i] - alphaIold) *oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T

        b2 = oS.b -Ei-oS.labelMat[i]*(oS.alphas[i] - alphaIold) *oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T

        if(0<oS.alphas[i]) and (oS.C>oS.alphas[i]):
            oS.b =b1
        elif (0<oS.alphas[j]) and (oS.C>oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1+b2)/2.0
        return 1
    else:
        return 0






if __name__=="__main__":
    dataArr,labelArr = loadDataSet('testSet.txt')

    # print(labelArr)
    b,alphas = smoSimple(dataArr,labelArr,0.6,0.001,40)
    # print(b,alphas)