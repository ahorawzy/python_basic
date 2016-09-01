'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator	#导入运算符模块
#operator.itemgetter(1)表示以第二个域的大小排序
from os import listdir
# 主要功能是列出给定目录的文件名

def classify0(inX, dataSet, labels, k):
	'''
	k近邻的核心算法
	'''
    dataSetSize = dataSet.shape[0]	#得到dataSet的行数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
	# 将inX利用np.tile拓展成dataSet一样的数据规模，然后元素相减，得到元素差
    sqDiffMat = diffMat**2
	# 求平方
    sqDistances = sqDiffMat.sum(axis=1)
	# 求行和
    distances = sqDistances**0.5
	# 开方
    sortedDistIndicies = distances.argsort()
	# np.argsort()方法返回数组值从小到大的索引值
    classCount={}  
	# 类别投票计数
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
		# 得到排序的第i个的标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
		# 在字典中查找，如果没有就默认为0，然后+1，赋值给classcount的第i个标签
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	# 字典的iter方法将所有项按列表的方式返回，每一项为键值对，operator.itemgetter(1)表示以第二个域的大小排序
    return sortedClassCount[0][0]

def createDataSet():
	'''
	创建数据集
	'''
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file2matrix(filename):
	'''
	将文本记录转换为Numpy的解析程序
	'''
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
	# 获得行数
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
	# 创建N行3列的Numpy零数组
    classLabelVector = []                       #prepare labels return   
	# 一个空的标签向量
    fr = open(filename)
    index = 0
	# 初始化一个行标记
    for line in fr.readlines():
	# 读取文件的每一行
        line = line.strip()
		# 去掉每一行的空格
        listFromLine = line.split('\t')
		# 按tab键分割，形成列表
        returnMat[index,:] = listFromLine[0:3]
		# Numpy数组的Index行，所有列赋予列表值
        classLabelVector.append(int(listFromLine[-1]))
		# 列表值得最后一个元素，转换为int型，追加到标签向量之后
        index += 1
		# 更新行标签
    return returnMat,classLabelVector
    
def autoNorm(dataSet):
	'''
	归一化特征值
	'''
    minVals = dataSet.min(0)
	# 寻找列的最小值
    maxVals = dataSet.max(0)
	# 寻找列的最大值
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
	# 按dataSet的大小，初始化一个零矩阵
    m = dataSet.shape[0]
	# 获得数组的行数
    normDataSet = dataSet - tile(minVals, (m,1))
	# 将minVals变换为和dataSet一样的数据规模，然后元素相减
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
	# 元素相除
    return normDataSet, ranges, minVals
   
def datingClassTest():
	'''
	测试代码
	'''
    hoRatio = 0.50      #hold out 10%
	# 定义测试样本所占比例
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
	# 得到行数
    numTestVecs = int(m*hoRatio)
	# 得到测试样本量
    errorCount = 0.0
	# 初始化错误数
    for i in range(numTestVecs):
	# 0到测试样本量的循环
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
		# 所测试的是测试样本量到最后一行
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount
    
def img2vector(filename):
	'''
	将32*32的数组转化为1*1024的向量
	'''
    returnVect = zeros((1,1024))
	# 先初始化一个零向量
    fr = open(filename)
	# 打开文件
    for i in range(32):
	# 遍历行
        lineStr = fr.readline()
		# 读取行，并存储
        for j in range(32):
		# 对于每个元素
            returnVect[0,32*i+j] = int(lineStr[j])
			# 存到returnVect当中
    return returnVect

def handwritingClassTest():
    hwLabels = []
	# 初始化一个空标签向量
    trainingFileList = listdir('trainingDigits')           #load the training set
	# 训练集目录赋予变量
    m = len(trainingFileList)
	# 获取文件个数
    trainingMat = zeros((m,1024))
	# 初始化一个m行，1024列的零矩阵
    for i in range(m):
	# 对于每个文件来说
        fileNameStr = trainingFileList[i]
		# 获得文件名字
        fileStr = fileNameStr.split('.')[0]     #take off .txt
		# 按.分割后，取第一部分，也就是去掉.txt
        classNumStr = int(fileStr.split('_')[0])
		# 从文件名中获得类标签，也就是按_分割后，第一部分
        hwLabels.append(classNumStr)
		# 将类标签存储到标签向量中
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
		# 将文件名传入img2vector函数，更新大矩阵
    testFileList = listdir('testDigits')        #iterate through the test set
	# 测试集目录赋予变量
    errorCount = 0.0
	# 初始化错误率
    mTest = len(testFileList)
	# 获得测试样本数
    for i in range(mTest):
	# 对于每一个测试样本
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))