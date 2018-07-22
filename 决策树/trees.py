from math import log
import operator
import treePlotter
# 1. 定义数据：*
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'N'],
               [0, 0, 0, 1, 'N'],
               [1, 0, 0, 0, 'Y'],
               [2, 1, 0, 0, 'Y'],
               [2, 2, 1, 0, 'Y'],
               [2, 2, 1, 1, 'N'],
               [1, 2, 1, 1, 'Y']]
    labels = ['outlook', 'temperature', 'humidity', 'windy']
    return dataSet, labels

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,"w")
    pickle.dumps(inputTree,fw)
    fw.close()

def grabTree(filename):
    import  pickle
    fr = open(filename)
    return pickle.load(fr)

#信息熵
def calcShannonEnt(dataset):
    numEntires = len(dataset)
    labCounts={}
    for featVec in dataset:
        currentLael = featVec[-1]
        if currentLael not in labCounts.keys():
            labCounts[currentLael] = 0
        labCounts[currentLael] +=1
    shannoEnt = 0
    for key in labCounts:
        prob = float(labCounts[key])/numEntires
        shannoEnt -= prob*log(prob,2)
    return shannoEnt

#划分数据
def splitDataSet(dataset,axis,value):
    retDataSet=[]
    for featVec in dataset:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的数据划分方式
def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0])-1
    baseEntropy = calcShannonEnt(dataset)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataset,i,value)
            prob = len(subDataSet)/float(len(dataset))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return  bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys() :classCount[vote] = 0
        classCount[vote]  +=1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
    return sortedClassCount

def createTree(dataset,labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0] )== len(classList):
        return classList[0]
    if len(dataset[0]) == 1:
        return  majorityCnt(classList)
    beatFeat = chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[beatFeat]
    myTree = {bestFeatLabel:{}}
    del (labels[beatFeat])
    featValues = [example[beatFeat] for example in dataset]
    uniquVals = set(featValues)
    for value in uniquVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataset,beatFeat,value),subLabels)
    return myTree

def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    print(firstStr)
    sencondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in sencondDict.keys():
        if testVec[featIndex] == key:
            if type(sencondDict[key]).__name__=='dict':
                classLabel = classify(sencondDict[key],featLabels,testVec)
            else:
                classLabel = sencondDict[key]
    return classLabel


dataset,lablesss = createDataSet()
label = lablesss
result = chooseBestFeatureToSplit(dataset)
desicionTree = createTree(dataset,label)
print(desicionTree)
treePlotter.createPlot(desicionTree)
result = classify(desicionTree,['outlook', 'temperature', 'humidity', 'windy'],[1,0,0])
print(result)

#预测隐形眼镜
fr = open("lenses.txt")
lenses = [inst.strip().split("\t") for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = createTree(lenses,lensesLabels)
print("lensesTree：  ",lensesTree)
treePlotter.createPlot(lensesTree)


