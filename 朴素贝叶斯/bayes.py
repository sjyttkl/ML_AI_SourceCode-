import numpy as np
import random
#词表到向量的转换函数
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
#s数据去重
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#把数据转成 向量形式。--->词集模型
def setOfword2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] =1
        else:
            print("the word :%s is not in my Vocabulary!"%word)
    return returnVec


#词袋模型
def bagOfWord2VecMN(vocaList,inputSet):
    returnVec = [0]*len(vocaList)
    for word in inputSet:
        if word in vocaList:
            returnVec[vocaList.index(word)] +=1
    return returnVec

#简单的朴素贝叶斯
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) /float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(0,numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num /p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    # print("=-==  ",vec2Classify*p1Vec)
    p1  = sum(vec2Classify*p1Vec) +np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else :
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfword2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(trainMat,listClasses)
    testEntry=['love', 'my', 'dalmation']
    thisDoc = np.array(setOfword2Vec(myVocabList,testEntry))
    print(testEntry,"classified as ",classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = np.array(setOfword2Vec(myVocabList,testEntry))
    print(testEntry,"classified as ",classifyNB(thisDoc,p0V,p1V,pAb))

def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList=[];classList=[];fullText=[]

    for i in range(1,26):
        # print(i)
        wordList = textParse(open('email/spam/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50));testSet =[]
    for i in range(0,10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainMat = [];trainClassess = []
    for docindex in trainingSet:
        trainMat.append(setOfword2Vec(vocabList,docList[docindex]))
        trainClassess.append(classList[docindex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClassess))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfword2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount +=1
            print("classification error", docList[docIndex])
    print("the error rate is :",float(errorCount)/len(testSet))


import feedparser
ny = feedparser.parse("http://newyork.craigslist.org/stp/index.rss")




if __name__ == "__main__":
    # listOPosts,listClasses = loadDataSet()
    # myVocabList = createVocabList(listOPosts)
    # print(myVocabList)
    # vector = setOfword2Vec(myVocabList,listOPosts[0])
    # print(vector)
    #
    # trainMat=[]
    # for postinDoc in listOPosts:
    #     trainMat.append(setOfword2Vec(myVocabList,postinDoc))
    # print(trainMat)
    # p0V,p1V,pAb = trainNB0(trainMat,listClasses)
    # print("p0V  ",p0V,"p1V  ",p1V,"pAb  ",pAb)
    # testingNB()
    # emailText = open("email/ham/6.txt").read()
    # print(textParse(emailText))
    spamTest()
