# logistic regression
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import xlrd
from pandas import DataFrame
from numpy.linalg import inv
import math
from sklearn.cross_validation import train_test_split
from numpy import array
from sklearn.model_selection import KFold

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=1000, fit_intercept=False, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) #/ y.size
            self.theta -= self.lr * gradient
            
            # if(i % 10000 == 0):
            #     z = np.dot(X, self.theta)
            #     h = self.__sigmoid(z)
            #     print('loss: {self.__loss(h, y)} \t')
        # print(self.theta)
        return self.theta
    
    def prob(self, X, w):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, w))

def main():
	data = pd.read_excel('data4.xlsx')
	var1 = data['row1']
	var2 = data['row2']
	var3 = data['row3']
	var4 = data['row4']
	var5 = data['row5']
	xmain = []
	for i in data.index:
		temp = []
		temp.append(var1[i])
		temp.append(var2[i])
		temp.append(var3[i])
		temp.append(var4[i])
		temp.append(var5[i])
		xmain.append(temp)
	xmain = np.array(xmain)
	# print(xmain)
	kfold = KFold(5, True, 1)
	for train, test in kfold.split(xmain):
		# xtra, xte, ytr, yte = train_test_split(xmain[:,:4], xmain[:, 4], train_size=0.6)
		# print(xmain[test])
		# print(ytr.T)
		xtra = xmain[train][:, :4]
		ytr = xmain[train][:, 4]
		xte = xmain[test][:, :4]
		yte = xmain[test][:, 4]
		xtr = []
		for i in range(len(ytr)):
			# temp = []
			# temp.append(xtra[i].tolist())
			temp = xtra[i].tolist()
			temp = np.append(temp, ytr[i].tolist()).tolist()
			xtr.append(temp)
		# print(xtr)
		x1 = []
		x2 = []
		x3 = []
		for xx in xtr:
			if xx[4]==1 :
				x1.append(xx[:4])
			elif xx[4]==2:
				x2.append(xx[:4])
			else :
				x3.append(xx[:4]) 

		x1 = np.array(x1)
		x2 = np.array(x2)
		x3 = np.array(x3)
		
		# 1 and 2 vs 3
		x3t=x3
		x = np.concatenate((x1, x2), axis=0)
		x = [np.append(xx.tolist(),0) for xx in x]
		x3t = [np.append(xx.tolist(),1) for xx in x3t]
		x = np.concatenate((x, x3t), axis=0)
		x = np.array(x)	
		# training
		model = LogisticRegression()
		wa = model.fit(x[:, :4], x[:, 4])
		# x.empty()

		# 1 and 3 vs 2
		x2t = x2
		xb = np.concatenate((x1, x3), axis=0)
		xb = [np.append(xx.tolist(),0) for xx in xb]
		x2t = [np.append(xx.tolist(),1) for xx in x2t]
		xb = np.concatenate((xb, x2t), axis=0)
		xb = np.array(xb)	
		# training
		wb = model.fit(xb[:, :4], xb[:, 4])

		# 3 and 2 vs 1
		xc = np.concatenate((x3, x2), axis=0)
		xc = [np.append(xx.tolist(),0) for xx in xc]
		x1 = [np.append(xx.tolist(),1) for xx in x1]
		xc = np.concatenate((xc, x1), axis=0)
		xc = np.array(xc)	
		# training
		wc = model.fit(xc[:, :4], xc[:, 4])
		# computing the accuracy
		y = []
		for xx in xte:
			a = model.prob(xx, wa)
			b = model.prob(xx, wb)
			c = model.prob(xx, wc)
			if a>b and a>c:
				y.append(3)
			elif b>a and b>c:
				y.append(2)
			else:
				y.append(1)

		y = np.array(y)

		acc = (yte == y).mean()
		print(acc)
	
if __name__ == "__main__":
	main()