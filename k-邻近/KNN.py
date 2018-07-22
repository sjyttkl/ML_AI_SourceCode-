from numpy import *
import operator
import numpy as np
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    # print(tile(inX,(dataSetSize,1)))
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDifMat = diffMat**2
    sqDistances = sqDifMat.sum(axis =1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        votIlabel = labels[sortedDistIndicies[i]]
        classCount[votIlabel] = classCount.get(votIlabel,0)+1
    #倒排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount

#数据准备
def file2matrix(filename):
    fr = open(filename,'r',encoding="utf-8")
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    print(returnMat)
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

#数值的归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

#测试分类器
def datingClassTest(filename):
    hoRatio = 0.10
    datingDatMat,datingLabels = file2matrix(filename)
    normMat,ranges,minVals = autoNorm(datingDatMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)#测试数据
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: ",classifierResult[0][0],"the real answer is  ",datingLabels[i])
        if(classifierResult[0][0] != datingLabels[i]) :
            errorCount +=1.0
    print("the total error rate is : " ,(errorCount/float(numTestVecs)))


#约会网站预测：
def classifyPerson():
   resultList = ['not at all','in small doses','in large doses']
   percentTats = float(input("percenttage of time spent playing video games?"))
   ffMiles = float(input("frequent flier miles earned per year?"))
   iceCream = float(input("liters of ice cream consumed per year?"))
   datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
   normMat, ranges, minVals = autoNorm(datingDataMat)
   inArr = np.array([ffMiles,percentTats,iceCream])
   classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
   print("You will probably like this perpon: ",resultList[classifierResult[0][0] - 1])


if __name__ == "__main__":
    group ,labels  = createDataSet()
    res = classify0([0,0],group,labels,3)
    print(res)

    datingDatMat,datingLables =  file2matrix("datingTestSet2.txt")
    print(datingDatMat,datingLables)
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDatMat[:,0],datingDatMat[:,1],15*array(datingLables),15*array(datingLables))
    #plt.show()

    normDataSet,ranges,minVals = autoNorm(datingDatMat)

    print(normDataSet,ranges,minVals)
    print("-----------knn--网站约会测试------------")
    datingClassTest("datingTestSet2.txt")
    print("-----------knn--网站约会预测------------")
    classifyPerson()