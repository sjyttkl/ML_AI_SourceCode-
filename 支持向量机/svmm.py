# -*- coding: UTF-8 -*-
"""
===============================================================
author：songdongdong
email：695492835@qq.com
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
    fXk= float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) +oS.b
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
        Ej=calcEk(oS,j)
    return j,Ej
def updateEk(oS,k):
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]

def innerL(i,oS):
    Ei = calcEk(oS,i)
    if((oS.labelMat[i]*Ei <-oS.tol) and (oS.alphas[i]<oS.C) or ((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>0))):
        j,Ej = selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold =oS.alphas[j].copy()
        if(oS.labelMat[i]!=oS.labelMat[j]):
            print(oS.alphas[i])

            L= np.maximum(0,oS.alphas[j] - oS.alphas[i]) #这里 注意了，不是max，是maximum
            H = np.minimum(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L=np.maximum(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H = np.minimum(oS.C,oS.alphas[j]+oS.alphas[i])
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



def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print ("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            # print(oS.alphas.A)
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print ("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print ("iteration number: %d" % iter)
    return oS.b,oS.alphas


def calcWs(alphas,dataArr,classLables):
    X = np.mat(dataArr)
    labelArr = np.mat(classLables).transpose()
    m,n = np.shape(X)
    w= np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelArr[i],X[i,:].T)
    return w

'''#######********************************
Kernel VErsions below
'''#######********************************

def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem --  That Kernel is not recognized')
    return K


class optStructKernel:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2))) #first column is valid flag
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def innerLKernel(i, oS):
        Ei = calcEkKernel(oS, i)
        if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C) or (
                (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0))):
            j, Ej = selectJ(i, oS, Ei)
            alphaIold = oS.alphas[i].copy()
            alphaJold = oS.alphas[j].copy()
            if (oS.labelMat[i] != oS.labelMat[j]):
                print(oS.alphas[i])

                L = np.maximum(0, oS.alphas[j] - oS.alphas[i])  # 这里 注意了，不是max，是maximum
                H = np.minimum(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
            else:
                L = np.maximum(0, oS.alphas[j] + oS.alphas[i] - oS.C)
                H = np.minimum(oS.C, oS.alphas[j] + oS.alphas[i])
            if L == H: print("L == H");return 0
            eta = 2.0  * oS.K[i, j].T - oS.K[i, i] - oS.K[j, j]
            if eta >= 0: print("eta>0");return 0;
            oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
            oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
            updateEk(oS, j)  # 更新误差缓存
            if (np.abs(oS.alphas[j] - alphaJold) < 0.00001):
                print("j no moving enough ");
                return 0
            oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
            updateEk(oS, i)  # 更新误差缓存

            b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,i] - oS.labelMat[
                j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]

            b2 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[
                j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]

            if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
                oS.b = b1
            elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
                oS.b = b2
            else:
                oS.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0
    #计算误差
def calcEkKernel(oS,k):
    fXk= float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[k,:].T) +oS.b
    Ek = fXk-float(oS.labelMat[k])
    return Ek


def smoPKernel(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStructKernel(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler,kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerLKernel(i,oS)
                print ("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            # print(oS.alphas.A)
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerLKernel(i,oS)
                print ("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print ("iteration number: %d" % iter)
    return oS.b,oS.alphas

def testRbfKerne(k1=1.3): #k1是蓉错率
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoPKernel(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    svInd=np.nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print ("there are %d Support Vectors" % np.shape(sVs)[0])
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1
    print ("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1
    print ("the test error rate is: %f" % (float(errorCount)/m))

##下面试svm识别数据
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)  # load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoPKernel(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = np.mat(dataArr);
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = np.mat(dataArr);
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))
if __name__=="__main__":
    # dataArr,labelArr = loadDataSet('testSet.txt')
    #
    # # print(labelArr)
    # # b,alphas = smoSimple(dataArr,labelArr,0.6,0.001,40)
    # b,alphas = smoP(dataArr,labelArr,0.6,0.001,40)
    # # print(b,alphas)
    # ws = calcWs(alphas,dataArr,labelArr)
    # print(ws)
    #
    # dataMat = np.mat(dataArr)
    # print("验证一下:  ",dataMat[0]*np.mat(ws)+b,labelArr[0])#验证一下
    # print("验证一下:  ", dataMat[1] * np.mat(ws) + b, labelArr[1])  # 验证一下
    # print("验证一下:  ", dataMat[2] * np.mat(ws) + b, labelArr[2])  # 验证一下


    # testRbfKerne(k1=1.20)

    testDigits(('rbf',20))