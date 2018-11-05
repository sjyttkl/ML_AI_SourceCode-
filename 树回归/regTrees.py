# -*- coding: UTF-8 -*-
"""
===============================================================
author：songdongdong
email：695492835@qq.com
date：2018.09.16
introduction:创建回归树
===============================================================
"""
class treenode:
    def __init__(self,feat,val,right,left):
        featureTpSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left

import numpy as np
def loadDataSet(fileName):
    dataMat= []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        # print(curLine)
        # python3不适用：fltLine = map(float,curLine) 修改为：
        fltLine = list(map(float, curLine))  # 将每行映射成浮点数，python3返回值改变，所以需要
        dataMat.append(fltLine)
    return dataMat

#分割
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]#这里好像不能加[0],书上写的有误
    # 下面原书代码报错 index 0 is out of bounds,使用上面两行代码
    # mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
    # mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
    return mat0,mat1


#是创建叶子节点的函数引用
def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])

#计算总方差，可以通过均方误差*数据中样本点的个数来计算
def regErr(dataSet):
    return np.var(dataSet[:,-1])*np.shape(dataSet)[0]

#构建函数
def createTree(dataSet,leafTye=regLeaf,errType=regErr,ops=(1,4)):
    feat,val = chooseBestSplit(dataSet,leafTye,errType,ops)#找到最佳切点
    if feat == None:return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataSet,feat,val)#根据最佳切分点开始切分
    retTree['left'] = createTree(lSet,leafTye,errType,ops)#构建左树
    retTree['right'] = createTree(rSet,leafTye,errType,ops)#构建右树
    return retTree

#用最佳方式切分数据集和生成相应的叶子节点
#这里最复杂，该函数的目标是找到数据集切分的最佳位置，它遍历所有的特征及其可能的取值来找到误差最小化的切分阈值
def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS = ops[0];tolN = ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None,leafType(dataSet)
    m,n = np.shape(dataSet)
    S = errType(dataSet)
    bestS= np.inf;bestIndex = 0;bestValue=0
    for featIndex in range(n-1):#取每个特征
        # for splitVal in set(dataSet[:,featIndex]): python3报错修改为下面
        for splitVal in set((dataSet[:,featIndex].T.tolist())[0]): #该特征中，每个特征值进行尝试切分
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            if(np.shape(mat0)[0] <tolN) or (np.shape(mat1)[0]<tolN): #tolN是最少样本数，如果比最少样本数还少，则不用切分了，继续下一个切分点
                continue
            newS= errType(mat0) +errType(mat1)#计算 总方差的和
            if newS < bestS:#如果比、新的总方差比最佳方差小，则更新最佳切分点。
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if(S-bestS)<tolS:#运行的误差下降值(效果提升调小，也就没有必要进行split)，如果小，则不进行切分，直接返回整个dataset
        return None,leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)#已经找到最佳切分的了，开始切分
    if(np.shape(mat0)[0]<tolN) or(np.shape(mat1)[0]<tolN):##tolN是最少样本数，如果比最少样本数还少，则不用切分了,Z直接返回整个值
        return None,leafType(dataSet)
    return bestIndex,bestValue


#测试变量是否是一颗树，换句话说判断是否是叶子节点
def isTree(obj):
    return (type(obj).__name__  =="dict")

#递归函数，从上到下，遍历树到叶子节点，如果找到两个叶子节点，则计算他们的平均树，该函数对树进行塌陷处理（即返回平均值)
def getMean(tree):
    if isTree(tree['right']):tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/2.0


def prune(tree, testData):
    if np.shape(testData)[0] == 0: return getMean(tree)  # if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):  # if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    # if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


#两个参数，待剪枝的树，以及需要测试的测试数据，
def prune2(tree,testData):
    if np.shape(testData)[0] == 0 :return getMean(tree) #没有测试数据，则对树进行塌陷处理
    #假设过拟合，对数据进行切分，对树进行剪枝
    if(isTree(tree['right']) or isTree(tree['left'])):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    #检查切分后的是树还是节点，如果是树进行切分，剪枝
    if isTree(tree['left']) :tree['left'] = prune(tree['left'],lSet)
    if isTree(tree['right']):tree['right'] = prune(tree['right'],rSet)
    #如果不是树了，是两个节点，则进行合并
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:,-1] - tree['left'],2)) + sum(np.power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(np.power(testData[:,-1] - treeMean,2))
        if errorMerge<errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree
#
#被其他函数调用，主要是把数据集格式化为目标变量Y和自变量X
def linearSolve(dataSet):
    m,n = np.shape(dataSet)
    X= np.mat(np.ones((m,n)));Y = np.mat(np.ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1];Y=dataSet[:,-1]
    xTx = X.T*X
    if np.linalg.det(xTx) == 0.0:#矩阵不能逆的时候，就报错
        raise NameError("this matrix is singular,cannot do inverse,try increasing the second vaue of ops")
    ws = xTx*(X.T*Y)
    return ws,X,Y

# 与上面的regLeaf类似，但是不需要负责生产叶子节点了，只需要返回回归系数就行了
def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws
#计算误差，与上面的regErr类似。这里计算平方误差
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X*ws
    return sum(np.power(Y-yHat,2))


def regTreeEval(model,inDat):
    return float(model)
def modelTreeEval(model,inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1,n+1)))
    X[:,1:n+1] = inDat
    return float(X*model)
def treeForeCast(tree,inData,modelEval=regTreeEval):
    if not isTree(tree):return modelEval(tree,inData)
    if inData[tree['spInd']] >tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)
def createForeCast(tree,testData,modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree,np.mat(testData[i]),modelEval)
    return  yHat












if __name__=="__main__":
    # testmat = np.mat(np.eye(4))
    # print(testmat)
    # mat0,mat1 = binSplitDataSet(testmat,1,0.5)
    # print(mat0,mat1)


    #==================================
    # myData = loadDataSet("ex2.txt")
    # myMat = np.mat(myData)
    # tree = createTree(myMat,ops=(0,1))
    # # print(tree)
    # myDataTest = loadDataSet("ex2test.txt")
    # myMat2Test = np.mat(myDataTest)
    # print(prune(tree,myMat2Test))

    #=======================模型树============
    # myMat2 = np.mat(loadDataSet('exp2.txt'))
    # print(createTree(myMat2,modelLeaf,modelErr,(1,10)))
    #==============树回归和标准回归的区别
    #----模型树测试--R2值，
    trainMat = np.mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = np.mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree = createTree(trainMat,ops=(1,20))
    yHat = createForeCast(myTree,testMat[:,0])
    print(np.corrcoef(yHat,testMat[:,1],rowvar=0)[0,1])
    #---回归树测试
    ws,X,Y = linearSolve(trainMat)
    print(ws)
    for i in range(np.shape(testMat)[0]):
        yHat[i] = testMat[i,0]*ws[1,0]+ws[0,0]
    print(np.corrcoef(yHat,testMat[:,1],rowvar=0)[0,1])
