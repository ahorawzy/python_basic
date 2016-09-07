'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''
from numpy import *

def loadSimpData():
# 产生训练数据
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
		# 数据集为矩阵形式
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
	# 类标签为列表形式
    return datMat,classLabels

def loadDataSet(fileName):      #general function to parse tab -delimited floats
# 解析文本数据
	numFeat = len(open(fileName).readline().split('\t')) #get number of fields
	# 对于类标签不是数字的，无法放进NumPy数组，可以先计算列数，确定列标签位置，再在解析的时候单独添加到列表中。
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
# 通过阈值比较对数据进行分类，就是简单的“决策树桩”
    retArray = ones((shape(dataMatrix)[0],1))
	# 产生一个m行1列的列向量
    if threshIneq == 'lt':
	# 如果表示为lt, less than
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
		# 那么小于这个阈值的被标记为-1
		# dataMatrix[:,dimen] <= threshVal返回每一行的布尔值，用于对行的筛选
    else:
	# 没有标记为lt, greater than
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
		# 大于阈值的被标记为1
    return retArray
    

def buildStump(dataArr,classLabels,D):
# 本函数用于向Adaboost算法做准备，构建一个决策树桩
# D为权重向量，是m行1列的向量，标识每条记录所占权重
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
	# 确保数据集为矩阵，类标签为列向量
    m,n = shape(dataMatrix)
	# 获得数据集的行数和列数，赋予m行n列
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
	# numSteps用于在特征的所有可能值上进行遍历
	# bestStump空字典，用于存储给定权重向量D时所得到的最佳单层决策树
	# 最佳类别预测初始化为m行1列的列向量
    minError = inf #init error sum, to +infinity
	# 最小错误率为正无穷
    for i in range(n):#loop over all dimensions
	# 遍历每一个特征
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
		# 找到此字段的最小值和最大值
        stepSize = (rangeMax-rangeMin)/numSteps
		# 通过范围和步数决定搜索步长
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            # 在范围（甚至之外一点）的区域内进行搜索
			for inequal in ['lt', 'gt']: #go over less than and greater than
			# 用以在lt和gt之间切换
                threshVal = (rangeMin + float(j) * stepSize)
				# 利用累增的步长确定此次搜索的值
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                # 利用决策树桩获得预测类
				errArr = mat(ones((m,1)))
				# 出错向量初始化为m行1列的列向量
                errArr[predictedVals == labelMat] = 0
				# 如果预测值和类标签值一致，则那一个记录的出错向量设为0
                weightedError = D.T*errArr  #calc total error multiplied by D
				# 将错误向量的每一个元素乘以权重向量的转置的每一个元素，得到加权错误
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
				# 如果加权错误小于最小错误
                    minError = weightedError
					# 最小错误为加权错误
                    bestClasEst = predictedVals.copy()
					# 最佳类别预测即为预测结果
                    bestStump['dim'] = i
					# 将最佳决策树桩的维度记录下来
                    bestStump['thresh'] = threshVal
					# 将字段的最佳阈值记录下来
                    bestStump['ineq'] = inequal
					# 将不等号的方向记录下来
    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
# adaBoostTrainDS的核心代码
# 输入参数包括：数据集、类别标签、迭代次数
# 可以将中间结果print出来，最后的成熟代码可以将print注释掉
    weakClassArr = []
	# 弱分类器分类向量
    m = shape(dataArr)[0]
	# m记录数据集的行数
    D = mat(ones((m,1))/m)   #init D to all equal
	# 初始化D为和为1的m行1列的列向量
    aggClassEst = mat(zeros((m,1)))
	# 集成累预测为全为1的列向量
    for i in range(numIt):
	# 对于每一次迭代
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
		#print "D:",D.T
		# 利用决策树桩将最佳树桩，错误和类预测记录下来
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        # 利用alpha的公式，通过错误率计算alpha
		# max(error,1e-16)保证没有除0错误
		bestStump['alpha'] = alpha  
        # 利用字典，记录alpha值，该字典记录了分类所需要的所有信息
		weakClassArr.append(bestStump)                  #store Stump Params in Array
        #print "classEst: ",classEst.T
        # 将最佳树桩的字典记录到弱分类向量的列表中，列表的元素可以是字典
		expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        #print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print "total error: ",errorRate
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        print aggClassEst
    return sign(aggClassEst)

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print "the Area Under the Curve is: ",ySum*xStep
