# -*- coding: UTF-8 -*-
"""
===============================================================
author：songdongdong
email：695492835@qq.com
date：2018
introduction:Apriori算法
===============================================================
"""
__author__ = "songdongdong"

import numpy as np
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    C1=[]
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset,C1)# 构成不变的集合项
#组织完整Apriori代码
def aprioriGen(Lk,k):#creates Ck
    retList = []
    lenLK = len(Lk)
    for i in range(lenLK):
        for j in range(i+1,lenLK):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            #前k-2项相同，将进行合并
            if L1==L2:
                retList.append(Lk[i]|Lk[j])
    return retList


def scanD(D,Ck,minSupport):
    ssCnt={}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt.keys():ssCnt[can]=1
                else:ssCnt[can]+=1
    numItems = float(len(D))
    retList=[]
    supportData={}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList,supportData

#扫描数据集，从CK 到 Lk
def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    C1 = list(C1)
    D = map(set, dataSet)
    D = list(D)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData
#关联规则生成
def generateRules(L,supportData,minConf=0.7):  #频繁集项列表，包含频繁项集的数据字典，最小可信度0.7
    bigRuleList = []
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]#遍历L中每一个频繁项集并对每个频繁项集创建只包含单个元素的项集开始规则构建过程
            if (i>1):#只获取有两个或更多的元素的集合
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(freqSet,H1,suppData,bigRuleList,minConf)
    return bigRuleList
#计算可信度的值
def calcConf(freSet,H,suppData,br1,minConf=0.7):
    pruneH=[] #存储规则的列表。
    for conseq in H:
        conf = suppData[freSet]/suppData[freSet-conseq]#一条规则P➞H的可信度定义为support(P | H)/support(P)，其中“|”表示P和H的并集。
        if conf >= minConf:
            print(freSet-conseq,'--->',conseq,'conf:',conf)
            br1.append((freSet-conseq,conseq,conf))
            pruneH.append(conseq)
    return pruneH

#合并,从最初的关联规则生成更多的了规则
def rulesFromConseq(freqSet,H,supportData,br1,minConf=0.7):
    m = len(H[0])
    if(len(freqSet)>(m+1)): #尝试进一步合并
        Hmp1 = aprioriGen(H,m+1) #创建Hm+1 条新候选规则 #合并
        Hmp1 = calcConf(freqSet,Hmp1,supportData,br1,minConf)
        if (len(Hmp1)>1):
            rulesFromConseq(freqSet,Hmp1,supportData,br1,minConf)




if __name__=="__main__":
    dataSet = loadDataSet()
    # C1 = createC1(dataSet=dataSet)
    # c1 = list(C1)
    # D = map(set,dataSet)
    # d = list(D)
    # L1,suppData0 = scanD(d,c1,0.5)
    # print(L1,suppData0)
    L,suppData = apriori(dataSet,minSupport=0.5)
    rules = generateRules(L,suppData,minConf=0.7)
    print(rules)
    #========发现毒蘑菇相似特征
    mushDataSet = [line.split() for line in open('mushroom.dat').readlines()]
    L,suppData=apriori(mushDataSet,minSupport=0.3)
    print(L)
    for item in L[3]:#L[2]
        print(item)
        if item.intersection('2'):#交集
            print(item)