import numpy as np
import operator
from os import listdir
"""
    time:2018/10/22
    author:hao chen
"""
# def createData():
#     simpleData = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 1.0]])
#     labels = ['A', 'A', 'B', 'B']
#
#     return simpleData, labels
# 读取数据
def  createData(filename):
    file = open(filename)
    arrayLine = file.readlines()
    m = len(arrayLine)
    returnMat = np.zeros((m, 3))
    classLabels = []
    index = 0
    for line in arrayLine:
        line = line.strip()
        listFline = line.split('\t')
        returnMat[index, :] = listFline[0:3]
        classLabels.append(listFline[-1])
        index = index+1
    return returnMat, classLabels
# 正则化
def autoNorm(dataSet):
    maxVals = np.max(dataSet, axis=0)
    minVals = np.min(dataSet, axis=0)
    ranges = maxVals-minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = np.shape(dataSet)[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1),)
    rangesMat = np.tile(ranges, (m, 1))
    normDataSet /= rangesMat
    return normDataSet, ranges, minVals
# 区分
def classify(inx, simpleData, labels, k):
    size = simpleData.shape[0]
    bias = np.tile(inx, (size, 1))-simpleData
    squbise = bias**2
    sumbias = squbise.sum(axis=1)
    distance = sumbias**0.5
    sortDistance = distance.argsort()

    classCount={}
    for i in range(k):
        votelabel = labels[sortDistance[i]]
        classCount[votelabel] = classCount.get(votelabel, 0)+1
    result = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return result[0][0]
# 测试
def datingClassTest():
    # 训练集与测试级比例
    hoRatio = 0.10
    returnMat, classLabels = createData('data/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(returnMat)
    m = normMat.shape[0]
    numTesVecs = int(m*hoRatio)
    errorNum = 0.0
    for i in range(numTesVecs):
        result = classify(returnMat[i, :], returnMat[numTesVecs: m], classLabels[numTesVecs: m], 3)
        print('the result of test is %s, the real answer is %s' % (result, classLabels[i]))
        if(result!=classLabels[i]):
            errorNum += 1.0
    print('the total error rate is : %f' % (errorNum/float(m)))

# 实际应用
def classifyPerson():
    resultList = ['not at all', 'is small does', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    inArr = np.array([percentTats, ffMiles, iceCream])
    returnMat, classLabels = createData("data/datingTestSet2.txt")
    autoMat, ranges, minVals = autoNorm(returnMat)
    classifyResult = int(classify((inArr-minVals)/ranges, autoMat, classLabels, 3))
    print("you will probably like this person:", resultList[classifyResult-1])

# 32*32图像转1*1024
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fil = open(filename)
    for i in range(32):
        lineStr = fil.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def getHandData(filename):
    hwLabels=[]
    fileList = listdir(filename)
    m = len(fileList)
    returnVec = np.zeros((m, 1024))
    for i in range(m):
        filenameStr = fileList[i]
        file = filenameStr.split('.')[0]
        fileNum=file.split('_')[0]
        hwLabels.append(int(fileNum))
        returnVec[i, :] = img2vector(filename+filenameStr)

    return returnVec, hwLabels

def handWritingTest():
    trainVec, trainLabels = getHandData('data/trainingDigits/')
    testVec, testLabels = getHandData('data/testDigits/')

    m = testVec.shape[0]
    errorNum = 0.0
    for i in range(m):
        classifyResult = classify(testVec[i, :], trainVec, trainLabels, 3)
        print("the value of test is %d,the real value is %d" % (classifyResult, trainLabels[i]))
        if(classifyResult != testLabels[i]):
            errorNum += 1.0
    print("the total number is %d, the error num is %f, the error rate is %f" % (m, errorNum, errorNum/m))

handWritingTest()