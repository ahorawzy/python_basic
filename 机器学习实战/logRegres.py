'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *


def loadDataSet():
# 便利函数，打开文件并读取
    dataMat = []; labelMat = []
	# 空数据集列表，空类标签集列表
    fr = open('testSet.txt')
	# 打开文件，并赋予文件变量
    for line in fr.readlines():
	# 读取文件的每一行，并迭代
        lineArr = line.strip().split()
		# 去除两端空格，并分割词汇，成为一个单词列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		# 数据集列表添加一行列表，将X0的值设为1.0，文件的前两个值分别是X1和X2
        labelMat.append(int(lineArr[2]))
		# 第三个值对应类别标签，追加到类标签集列表中
    return dataMat,labelMat

def sigmoid(inX):
# 定义一个sigmoid函数
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
# 锑度上升算法的核心代码
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
	# 将NumPy二维数组转化为NumPy矩阵
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
	# 行向量转化为列向量，并原地转置
    m,n = shape(dataMatrix)
	# 获得矩阵的维度
    alpha = 0.001
	# 设置步长
    maxCycles = 500
	# 设置最大迭代次数
    weights = ones((n,1))
	# 初始化权重,ones((4,1))创造一个4*1的全1矩阵
    for k in range(maxCycles):              #heavy on matrix operations
	# 在最大迭代次数内
        h = sigmoid(dataMatrix*weights)     #matrix mult
		# 这里的乘法是矩阵乘法，h不是一个数，而是一个列向量，长度为100
        error = (labelMat - h)              #vector subtraction
		# 计算偏差，error也是列向量
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
		# 按差值的方向调整回归系数
		# dataMatrix是m*n的，转置为n*m的，error是m*1的，矩阵乘完是n*1的，正好加入原始权重中，进行调整
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
# 没有矩阵转化过程，所有变量的数据类型都是NumPy数组
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
	# 遍历数据集的每一行
        h = sigmoid(sum(dataMatrix[i]*weights))
		# h是数字，而不是向量
        error = classLabels[i] - h
		# error也是数字，而不是向量
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))
        