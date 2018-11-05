# -*- coding: UTF-8 -*-
"""
===============================================================
author：songdongdong
email：695492835@qq.com
date：2018.09.23
introduction:svd
===============================================================
"""
__author__ = "songdongdong"

import numpy as np
def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

#相似度计算
def ecludSim(inA,inB):
    return 1.0/(1.0+np.linalg.norm(inA-inB))

#皮尔逊相关系数范围是-1,到1，通过加0.5，归一化到0,1
def pearsSim(inA,inB):
    if len(inA)<3:return 1.0
    return 0.5+0.5*np.corrcoef(inA,inB,rowvar=False)[0][1]
#余玄相似度，如果夹角为90则相似度为0
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = np.linalg.norm(inA)*np.linalg.norm(inB)
    return 0.5+0.5*(num/denom)
#标准的协同过滤的推荐系统,，矩阵，用户编号物品编号，相似度计算方法，某个物品
#用来计算在给定相似度计算方法的条件下，用户对物品的估计评分值
def standEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0;ratSimTotal=0.0
    for j in range(n):
        userRating=dataMat[user,j]#用户评级的其他商品
        if userRating == 0 :continue
        overLap = np.nonzero(np.logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]#寻找两个都被用户评价过的物品
        if len(overLap) == 0 : similarity=0
        else:
            similarity=simMeas(dataMat[overLap,item],dataMat[overLap,j])#这其实就是物品的相似度
            print("the %d and %d similarity is :  %f"%(item,j,similarity))
            simTotal+=similarity
            ratSimTotal +=similarity*userRating#每次计算需要和当前评分乘积
    if simTotal ==0: return 0
    else:
        return ratSimTotal/simTotal#相似度进行规一划，把最后的评分控制在0-5之间
#SVD 减少特征空间的推荐算法
def svdEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = np.linalg.svd(dataMat)
    Sig4 = np.mat(np.eye(4)*Sigma[:4]) #这里的4 的确定，是因为下面已经判断过利用启发式的方式判断了 前3维可以代表90%的数据
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  #create transformed items利用U将物品转移到低维空间
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T, xformedItems[j,:].T)
        print ('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal
#矩阵，用户，前N个，相似度计算方法，估计方法
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=svdEst):
    unratedItems = np.nonzero(dataMat[user,:].A==0)[1]#find unrated items寻找未评级的物品，也就是矩阵的列
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print (1)
            else: print (0,)
        print ('')

def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = np.mat(myl)
    print ("****original matrix******")
    printMat(myMat, thresh)
    U,Sigma,VT = np.linalg.svd(myMat)
    SigRecon = np.mat(np.zeros((numSV, numSV)))
    for k in range(numSV):#construct diagonal matrix from vector
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print ("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)


if __name__=="__main__":
    # Data = loadExData()
    # U,Sigma,VT = np.linalg.svd(Data)
    # print(np.shape(U))
    # print(Sigma)
    #师徒重构原始矩阵
    # sig3 =np.mat([[Sigma[0],0,0],[0,Sigma[1],0],[0,0,Sigma[2]]])
    # print(U[:,:3]*sig3*VT[:3,:])

    #=====开始计算======================
    myMat = np.mat(loadExData())
    # print(ecludSim(myMat[:,0],myMat[:,4]))
    # print(ecludSim(myMat[:, 0], myMat[:,0]))
    # print(cosSim(myMat[:, 0], myMat[:, 4]))
    # print(cosSim(myMat[:, 0], myMat[:, 0]))
    # print(pearsSim(myMat[:, 0], myMat[:, 0]))
    # print(pearsSim(myMat[:, 0], myMat[:, 4]))
    # ====================================基于协同过滤的推荐系统=========================
    myMat = np.mat(loadExData())
    myMat[0,1] = myMat[0,0]=myMat[1,0]=myMat[2,0] = 4
    myMat[3,3]=2
    print(recommend(myMat,2))
    print(recommend(myMat,2,simMeas=ecludSim))
    print(recommend(myMat,2,simMeas=pearsSim))
    # ====================利用SVD提高推荐的效果======================
    #现实中的矩阵会很稀疏
    U,sigma,VT = np.linalg.svd(np.mat(loadExData2()))
    print(sigma)
    #下面是确定到底需要保留前面多少个---启发式方法
    Sig2 = sigma**2
    print(sum(Sig2))
    #计算总量的90%
    print(sum(Sig2)*0.9)
    print(Sig2[:2],sum(Sig2[:2]))#计算前两个元素所包含的能量 该值低于90%
    print(Sig2[:3],sum(Sig2[:3]))#该值高于90%，这样就可以把11维的数据转成一个3维的数据
    #======================图像上的压缩================
    imgCompress(2)


