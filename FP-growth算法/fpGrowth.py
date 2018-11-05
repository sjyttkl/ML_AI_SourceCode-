# -*- coding: UTF-8 -*-
"""
===============================================================
author：songdongdong
email：695492835@qq.com
date：2018.09.22
introduction: Fp-growth 算法
===============================================================
"""
#自定义数据结构
class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name=nameValue #名称
        self.count=numOccur#计数
        self.nodeLink=None#链接相似的元素项
        self.parent=parentNode#父节点
        self.children={}#孩子节点
    def inc(self,numOccur):
        self.count+=numOccur #增加定值
    #将树以文本形式显示
    def disp(self,ind=1):
        print(" "*ind,self.name," ",self.count)
        for child in self.children.values():
            child.disp(ind+1)
def createTree(dataSet,minSup=1):
    headerTable ={}#这是表头指针
    for trans in dataSet:
        for item in trans:
            headerTable[item]=headerTable.get(item,0)+dataSet[trans]
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del headerTable[k]  #移除不满足最小支持度的元素项
    freqItemSet = set(headerTable.keys())
    #如果没有元素满足，则退出
    if len(freqItemSet) == 0: return None,None
    for k in headerTable:
        headerTable[k] = [headerTable[k],None]
    retTree  = treeNode('Null Set',1,None)
    for tranSet,count in dataSet.items():
        localD={}
        #根据全局频率对每个事物中的元素进行排序---还必须是频繁的
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]#只有对频繁的排序才有意义
        if len(localD)>0:
            orderedItems = [v[0] for v in sorted(localD.items(),key=lambda p:p[1],reverse=True)]
            updateTree(orderedItems,retTree,headerTable,count)
    return retTree,headerTable

def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:#check if orderedItems[0] in retTree.children
        inTree.children[items[0]].inc(count) #incrament count
    else:   #add items[0] to inTree.children
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None: #update header table
            headerTable[items[0]][1] = inTree.children[items[0]] #这里是更新表头指针
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:#call updateTree() with remaining ordered items#处理第二个
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


#这里其实就是一个链表
def updateHeader(nodeToTest, targetNode):  # this version does not use recursion
    while (nodeToTest.nodeLink != None):  # Do not use recursion to traverse a linked list!
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

#构建FP树
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


def ascendTree(leafNode, prefixPath):  # ascends from leaf node to root
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):  # treeNode comes from header table
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats
if __name__=="__main__":
    # rootNode = treeNode("pyramid",9,None)
    # rootNode.children['eye'] = treeNode("eye",13,None)
    # print(rootNode.disp())
    # ==============构建FP树===========
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    print(initSet)
    myFPtree,myHeaderTab = createTree(initSet,3)
    myFPtree.disp()