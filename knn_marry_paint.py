import  numpy as np
import matplotlib.pyplot as plt
import operator
"""
    time:2018/10/22
    author:hao chen
"""
def autoNorm(dataSet):
    maxVals = np.max(dataSet, axis=0)
    minVals = np.min(dataSet, axis=0)
    ranges = maxVals-minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = np.shape(dataSet)[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1),)
    rangesMat = np.tile(ranges, (m, 1))
    normDataSet /= rangesMat
    return normDataSet

def createData(filename):
    data = open(filename)
    arrayOload = data.readlines()
    loadSize = len(arrayOload)
    returnMat = np.zeros((loadSize, 3))
    classLabel = []
    load = np.zeros((loadSize, 4))
    index = 0
    for line in arrayOload:
        line = line.strip()
        listFrolin = line.split("\t")
        load[index] = listFrolin
        index += 1

    load = autoNorm(load)
    m = load.shape[0]
    index = 0
    for i in range(m):
        listFrolin = load[i]
        returnMat[index, :] = listFrolin[0:3]
        classLabel.append(float(listFrolin[-1]))
        index += 1
    return returnMat, classLabel

def paintResult(resultMat, classLabel):
    flg = plt.figure()
    ax = flg.add_subplot(111)
    ax.set_title("网约网站数据")
    ax.set_xlabel("玩游戏百分比")
    ax.set_ylabel("消耗冰淇淋公斤数")
    ax.scatter(resultMat[:, 1], resultMat[:, 2])
    #保存图片
    # plt.savefig("img/marry.png")
    plt.show()

simpleData, simpleLabels = createData("data/datingTestSet2.txt")


