# _*_ coding:utf-8 _*_
import os
from numpy import *
import operator

def createDataSet():
    group=array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels='A', 'A', 'B', 'B'
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffM = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffM = diffM**2
    sqDist = sqDiffM.sum(axis=1)
    dist = sqDist**(0.5)
    distOrderIndex = dist.argsort()

    classCount = {}

    for i in range(k):
        voteIlabel = labels[distOrderIndex[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=lambda x: x[1], reverse=True)
    #sortedClassCount = sorted(classCount.iteritems(), key=operater.itemgetter(1),reverse=True)

    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename, 'r')
    arrayOflines = fr.readlines()
    numOflines = len(arrayOflines)
    dataMatrix = zeros((numOflines,3))
    index = 0
    labelVector = []
    for line in arrayOflines:
        listline = line.strip()
        listFromline = listline.split('\t')
        dataMatrix[index,:] = listFromline[0:3]
        labelVector.append(listFromline[-1])
        index+=1

    return dataMatrix, labelVector
#文件转换为矩阵


def autonorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normedDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normedDataSet = dataSet - tile(minVals, (m,1))
    normedDataSet = normedDataSet / tile(ranges, (m,1))
    return normedDataSet, ranges, minVals
#数据的归一化


def datingClassTest():
    dataSet, labels = file2matrix('datingTestSet.txt')
    normData, ranges, minVals = autonorm(dataSet)
    m = normData.shape[0]
    hOrate = 0.5
    errNum = 0.0
    numTest = int(m*hOrate)
    for i in range(numTest):
        classifyResult = classify0(normData[i,:], normData[numTest:m,:], labels[numTest:m], 8)
        #print "The classifier come back with %s, and the real answer is %s" %(classifyResult, labels[i])
        if classifyResult != labels[i]:
            errNum+=1.
    print "The total rate of error is %.2f%%." %((errNum/float(numTest))*100.)

def classifyPerson():
    patterns = {'largeDoses':u'极具魅力', 'smallDoses':u'魅力一般', 'didntLike':u'不喜欢'}
    gameTime = float(raw_input('Percentage of time spend on video games: '))
    flightMiles = float(raw_input('frequent flied miles earned per year: '))
    iceCream = float(raw_input('liter of icecream consumed per month: '))
    dataSet, labels = file2matrix('datingTestSet.txt')
    normData, ranges, minVals = autonorm(dataSet)
    inX = array([flightMiles, gameTime, iceCream])
    normInX = (inX - minVals)/ranges
    classifyResult = classify0(normInX, normData, labels, 8)
    print patterns[classifyResult]


def img2Vec(filename):
    fr = open(filename, 'r')
    returnVecs = zeros((1,1024))
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVecs[0, 32*i+j] = int(lineStr[j])
    return returnVecs


def handwritingClassTest():
    frname = os.listdir('trainingDigits')
    m = len(frname)
    hwLabels = []
    trainImgs = zeros((m,1024))
    for i in range(m):
        hwLabels.append(int((((frname[i].split('.'))[0]).split('_'))[0]))
        trainImgs[i,:] = img2Vec('trainingDigits/%s'%frname[i])
    #训练数据格式化

    frTestName = os.listdir('testDigits')
    n = len(frTestName)
    errNum = 0.
    testLabels = []
    for i in range(n):
        testLabels = int((((frTestName[i].split('.'))[0]).split('_'))[0])
        testData = img2Vec('testDigits/%s'%frTestName[i])
        testResult = classify0(testData, trainImgs, hwLabels, 3)
        if testResult != testLabels:
            errNum+=1
    print 'The total error rate is %.2f%%'%((errNum/float(n))*100.)
