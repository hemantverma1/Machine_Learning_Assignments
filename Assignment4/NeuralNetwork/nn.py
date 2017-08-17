import numpy
import sys
import csv
from math import sqrt

def sigmoid(x):
	return float(1.0/(1.0 + numpy.exp(-x)))

def differentiationSigmoid(x):
	return sigmoid(x)*(1.0-sigmoid(x))

def net(W, X):		#net = W^t*X
	n = W.shape[1]
	Net = numpy.zeros((n,1), dtype=numpy.float64)
	Net = numpy.add(Net, numpy.dot(W.transpose(), X))
	return Net

def sensitivityK(TkVec, ZkVec, netKVec):	#sensitivity at layer k (O/P layer)
	#TkVec is a vector of 3 elements
	#ZkVec is a vector of 3 elements
	#netKVec is a vector of 3 elements
	sensitivityK_Vec = numpy.subtract(TkVec, ZkVec)
	for i in xrange(sensitivityK_Vec.shape[0]):
		sensitivityK_Vec[i][0] *= differentiationSigmoid(netKVec[i][0])
	return sensitivityK_Vec
	
def sensitivityJ(WkjVec, sensitivityK_Vec, netJVec):	#sensitivity at layer j (Hidden layer)
	#WkjVec is a 2D matrix of size (nH+1 X 3) - last element is Wo
	#sensitivityK_Vec is a vector of 3 elements
	#netJVec is a vector of nH elements
	sensitivityJ_Vec = numpy.zeros((len(netJVec)+1,1), dtype=numpy.float64)
	sensitivityJ_Vec = numpy.add(sensitivityJ_Vec, numpy.dot(WkjVec, sensitivityK_Vec))
	for j in xrange(netJVec.shape[0]):
		sensitivityJ_Vec[j][0] *= differentiationSigmoid(netJVec[j][0])
	return sensitivityJ_Vec

def neuralNet3Layer(WkjVec, WjiVec, errorThreshold, eita, TkMat, X_Mat, Xtest_Mat, targetDigitVec): 	
	#implements on-line learning algo(back propagation) for 3 layer neural net
	#WkjVec is a 2D matrix of size (nH+1 X 3)
	#WjiVec is a 2D matrix of size (ni+1 X nH)
	#TkMat is a matrix of target value vectors
	#X_Mat is a matrix of augmented column vectors - 1 augmented in the end of the vector
	error = 9999.0;			#big value so that loop runs at least once
	i = 0;
	n = X_Mat.shape[1]		#no of feature vectors
	count = 0
	while(error > errorThreshold):
		X = X_Mat[:, i]
		X = X.reshape((len(X), 1))	#real column vector
		TkVec = TkMat[:, i]
		TkVec = TkVec.reshape((len(TkVec), 1))
		netJVec = net(WjiVec, X)
		YjVec = calculateYj(WjiVec, X)
		YjVecAugmented = numpy.append(YjVec, numpy.array([1]).reshape((1,1)), axis=0)
		netKVec = net(WkjVec, YjVecAugmented)
		ZkVec = calculateZk(WkjVec, YjVecAugmented)
		delK_Vec = sensitivityK(TkVec, ZkVec, netKVec)
		delJ_Vec = sensitivityJ(WkjVec, delK_Vec, netJVec)
		WkjVec = numpy.add(WkjVec, numpy.dot(YjVecAugmented, delK_Vec.transpose())*eita)
		tmpVec = numpy.dot(X, delJ_Vec.transpose())
		WjiVec = numpy.add(WjiVec, tmpVec[:, :-1]*eita)
		#calculate the error
		YjVec = calculateYj(WjiVec, X)	#now use new updated WjiVec
		YjVecAugmented = numpy.append(YjVec, numpy.array([1]).reshape((1,1)), axis=0)
		ZkVec = calculateZk(WkjVec,	YjVecAugmented)	#now use new updated WkjVec	
		errorVecNorm = numpy.linalg.norm(numpy.subtract(TkVec, ZkVec))
		error = 0.5*pow(errorVecNorm, 2)
		'''
		error = calculateError(WjiVec, WkjVec, TkMat, X_Mat)
		'''
		if count%500==0:
			print "count ", count, " error ", error
		i = (i+1)%n
		count += 1
	print "Accuracy: ", test(WjiVec, WkjVec, targetDigitVec, Xtest_Mat)

def calculateError(WjiVec, WkjVec, TkMat, X_Mat):
	error = 0.0
	for i in xrange(X_Mat.shape[1]):
		X = X_Mat[:, i]
		X = X.reshape((len(X), 1))	#real column vector
		TkVec = TkMat[:, i]
		TkVec = TkVec.reshape((len(TkVec), 1))
		YjVec = calculateYj(WjiVec, X)	#now use new updated WjiVec
		YjVecAugmented = numpy.append(YjVec, numpy.array([1]).reshape((1,1)), axis=0)
		ZkVec = calculateZk(WkjVec,	YjVecAugmented)	#now use new updated WkjVec	
		errorVecNorm = numpy.linalg.norm(numpy.subtract(TkVec, ZkVec))
		error += 0.5*pow(errorVecNorm, 2)
	return error
		
def calculateZk(WkjVec, YjVec):
	return calculateYj(WkjVec, YjVec)

def calculateYj(WjiVec, X):
	nH = WjiVec.shape[1]
	YjVec = numpy.zeros((nH,1), dtype=numpy.float64)
	YjVec = numpy.add(YjVec, numpy.dot(WjiVec.transpose(), X))
	for i in xrange(nH):
		YjVec[i][0] = sigmoid(YjVec[i][0])
	return YjVec

def test(WjiVec, WkjVec, targetDigitVec, X_Mat):
	accuracy = 0.0
	for i in xrange(X_Mat.shape[1]):
		X = X_Mat[:, i]
		X = X.reshape((len(X), 1))	#real column vector
		YjVec = calculateYj(WjiVec, X)
		YjVecAugmented = numpy.append(YjVec, numpy.array([1]).reshape((1,1)), axis=0)
		ZkVec = calculateZk(WkjVec, YjVecAugmented)
		print ZkVec
		ans = numpy.argmax(ZkVec)
		print "ans given", ans
		print "true answer", targetDigitVec[i]
		if(targetDigitVec[i]==ans):
			accuracy += 1
	return accuracy/X_Mat.shape[1]	
	

if __name__ == "__main__":
	eita = 0.05
	errorThreshold = 0.001
	nH = int(input("nH: "))
	c = 3
	ni = 64
	#WkjVec is a 2D matrix of size (nH+1 X 3)
	#WjiVec is a 2D matrix of size (ni+1 X nH)
	#TkVec is a vector of 3 elements
	#X_Mat is a matrix of augmented column vectors - 1 augmented in the end of the vector
	wji = numpy.random.randn(ni+1, nH)
	wkj = numpy.random.randn(nH+1, c)
	for i in range(len(wji)):
		for j in range(len(wji[i])):
			wji[i][j] *= (1/sqrt(ni))
	for i in range(len(wkj)):
		for j in range(len(wkj[i])):
			wkj[i][j] *= (1/sqrt(nH))
	X_Mat = numpy.zeros((ni+1,1))
	TkMat = numpy.zeros((c,1))
	with open('optdigits.tra.csv','rb') as f1:
		reader = csv.reader(f1)
		for row in reader:
			#make a column vector of first 64 features
			row = [int(x) for x in row]
			X =  numpy.append(numpy.asarray(row[:-1]), 1).reshape((ni+1, 1))
			X_Mat = numpy.hstack((X_Mat, X))
			targetDigit = row[-1]
			Tk = numpy.zeros((c,1))
			Tk[targetDigit] = 1
			TkMat = numpy.hstack((TkMat, Tk))
	X_Mat = X_Mat[:, 1:]
	TkMat = TkMat[:, 1:]
	#Testing data
	Xtest_Mat = numpy.zeros((ni+1,1))
	targetDigitVec = []
	with open('optdigits.tes.csv','rb') as f2:
		reader = csv.reader(f2)
		for row in reader:
			row = [int(x) for x in row]
			X =  numpy.append(numpy.asarray(row[:-1]), 1).reshape((ni+1, 1))
			Xtest_Mat = numpy.hstack((Xtest_Mat, X))
			targetDigitVec.append(row[-1])
		Xtest_Mat = Xtest_Mat[:, 1:]
	neuralNet3Layer(wkj, wji, errorThreshold, eita, TkMat, X_Mat, Xtest_Mat, targetDigitVec)
			
	'''
	eita = 0.5
	errorThreshold = 0.001
	WjiVec = numpy.array([[0.15, 0.25], [0.20, 0.30], [0.35, 0.35]])
	WkjVec = numpy.array([[0.40, 0.5], [0.45, 0.55], [0.60, 0.60]])
	X_Mat = numpy.array([[0.05], [0.10], [1]])
	TkVec = numpy.array([[0.01], [0.99]])
	neuralNet3Layer(WkjVec, WjiVec, errorThreshold, eita, TkVec, X_Mat)
	'''

