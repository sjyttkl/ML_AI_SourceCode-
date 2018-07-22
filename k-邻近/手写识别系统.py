import KNN
import numpy as np
from os import listdir
#把图像数字转换成向量形式
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr =  open(filename)
    for i in range(0,32):
        lineStr = fr.readline()
        for j in range(0,32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#手写识别系统
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir("trainingDigits")
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(0,m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector("trainingDigits/"+fileNameStr)
    testFileList = listdir("testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(0,mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest= img2vector("testDigits/" + fileNameStr)
        classiferResult = KNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: "+str(classiferResult[0][0])+",the real answer is: "+str(classNumStr))
        if classiferResult[0][0] != classNumStr:
            errorCount += 1.0
    print("\nthe total number od errors is: "+ str(errorCount))
    print("\nthe total error rate is: ",(errorCount / float(mTest)))






if __name__ == "__main__":
    testVector = img2vector("testDigits/0_0.txt")
    print(testVector)
    handwritingClassTest()