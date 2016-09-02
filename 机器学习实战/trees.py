'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington

递归结束的条件是：遍历完所有划分数据集的属性，或每个分支下的所有实例都具有相同的分类
'''
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):
	"计算熵"
    numEntries = len(dataSet)
	# 获得实例总数
    labelCounts = {}
	# 初始化类计数的空字典
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
		# 获取最后一个元素作为当前类标签
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
		# 如果当前类标签不在类标签字典里，，拓展字典，并添加类标签为0
        labelCounts[currentLabel] += 1
		# 不管在不在字典里，当前类标签的计数+1
    shannonEnt = 0.0
	# 初始化香农熵
    for key in labelCounts:
	# 对于类计数中的每个键，即类
        prob = float(labelCounts[key])/numEntries
		# 概率等于类计数/实例总数，变为浮点型
        shannonEnt -= prob * log(prob,2) #log base 2
		# 香农熵的计算公式：负的log以2为底的概率的和
    return shannonEnt
    
def splitDataSet(dataSet, axis, value):
	'''
	按给定特征划分数据集
	其中输入：
	dataSet	待划分数据集
	axis	划分数据集的特征
	value	需要返回的特征的值
	'''
    retDataSet = []
	# 初始化一个新列表
    for featVec in dataSet:
	# 遍历每个实例
        if featVec[axis] == value:
		# 如果实例的特征等于value
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
			# 上面两行将选择的axis特征抽取出来
            retDataSet.append(reducedFeatVec)
			# 添加到新列表中
    return retDataSet
    
def chooseBestFeatureToSplit(dataSet):
	'''
	选择最好数据集划分方式
	'''
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
	# len(dataset[0])是计算数据集的列数，最后一列是标签列
    baseEntropy = calcShannonEnt(dataSet)
	# 调用熵计算公式，算数据集的初始熵
    bestInfoGain = 0.0; bestFeature = -1
	# 初始化最大信息增益为0，初始化最佳特征为-1
    for i in range(numFeatures):        #iterate over all the features
	# 遍历每一个特征
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        # 遍历数据集的每一行，将特征为i的值记录在一个列表中
		uniqueVals = set(featList)       #get a set of unique values
		# 过滤掉重复元素，形成特征i的值的集合
        newEntropy = 0.0
		# 初始化新的熵为0
        for value in uniqueVals:
		# 遍历值集合中的每个值
            subDataSet = splitDataSet(dataSet, i, value)
			# 划分子数据集
            prob = len(subDataSet)/float(len(dataSet))
			# 子数据集发生的概率按频率来算
            newEntropy += prob * calcShannonEnt(subDataSet)   
			# 求新熵的期望值
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        # 计算信息增益，信息增益就是熵的减少
		if (infoGain > bestInfoGain):       #compare this to the best gain so far
            # 如果此划分的信息增益大于最大信息增益
			bestInfoGain = infoGain         #if better than current best, set to best
			# 最大信息增益等于此划分的信息增益
            bestFeature = i
			# 最佳特征就是特征i
    return bestFeature                      #returns an integer

def majorityCnt(classList):
'''
多数表决投票
'''
    classCount={}
	# 设置一个空字典
    for vote in classList:
	# 对于类列表中记录的每一票
        if vote not in classCount.keys(): classCount[vote] = 0
		# 如果那一票不在字典中，那么设置这一票的频数为0
        classCount[vote] += 1
		# 这一票的频数+1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	# 字典的iter方法将所有项按列表的方式返回，每一项为键值对，operator.itemgetter(1)表示以第二个域的大小排序
    return sortedClassCount[0][0]
	# 返回频数最多的键值对元组中的第一维，即键

def createTree(dataSet,labels):
'''
创建树的代码
'''
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
