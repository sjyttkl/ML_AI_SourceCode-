import  numpy as np
import random
def loadDataSet():
    dataMat=[]
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    alaph = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(0,maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat-h)
        weights = weights+alaph *dataMatrix.transpose()*error
    return weights

def plotBestFit(wei):
    import matplotlib.pyplot as plt
    dataMat,labelmat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2=[]
    for i in range(0,n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = np.arange(-3.0,3.0,0.1)
    # print(x)
    # y = (-wei[0][0]-wei[1][0]*x)/wei[2][0]
    y = (-wei[0] - wei[1] * x) / wei[2]
    ax.plot(x,y)
    plt.xlabel('X!')
    plt.ylabel('X2')
    plt.show()
#s随机梯度上升算法
def stocGradAscent0(dataMatrix,classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(0,m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i]-h
        # print(dataMatrix[i])
        weights = weights +alpha*error *dataMatrix[i]
    return weights
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(0,numIter):
        dataIndex = list(range(0,m))
        for i in range(0,m):
            alpha = 4/(1.0+j+i) +0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] -h
            weights=weights+ alpha*error*dataMatrix[randIndex]
            del dataIndex[randIndex]
    return weights

#--------------从疝气病预测病马的死亡率------
def classifyVecotr(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob >0.5:
        return 1.0
    else:
        return 0.0
def coliTest():
    frTrain = open("horseColicTraining.txt")
    frTest = open("horseColicTest.txt")
    trainingSet = []
    trainingLabels=[]
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(0,21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights= stocGradAscent1(np.array(trainingSet),trainingLabels,500)
    errorCount = 0;numTestVec=0.0
    for line in frTest.readlines():
        numTestVec +=1.0
        currLine = line.strip().split('\t')
        lineArr=[]
        for i in range(0,21):
            lineArr.append(float(currLine[i]))
        if int(classifyVecotr(np.array(lineArr),trainWeights))!= int(currLine[21]):
            errorCount +=1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is :%f"%errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(0, numTests):
        errorSum += coliTest()
    print("after %d iteratons the average error rate is :%f" % (numTests, errorSum / float(numTests)))



if __name__ =="__main__":
    datMat,labelMat = loadDataSet()
    # print(datMat,labelMat)
    import  time
    #批处理
    # start = time.time()
    weights = gradAscent(np.array(datMat),labelMat)
    # plotBestFit(weights.getA())
    print(weights)
    # print(time.time()-start)

    #随机梯度上升，在线学习算法
    # start = time.time()
    weights = stocGradAscent0(np.array(datMat),labelMat)
    print(weights)
    # plotBestFit(weights)
    # print(time.time() - start)

    #随机梯度上升，在线学习算法
    start = time.time()
    weights = stocGradAscent1(np.array(datMat), labelMat)
    print(weights)
    # plotBestFit(weights)
    # print(time.time() - start)

    #从疝气病预测病马的死亡率
    multiTest()

