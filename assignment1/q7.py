# logistic regression
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import xlrd
from pandas import DataFrame
from numpy.linalg import inv
import math
from sklearn.cross_validation import train_test_split

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=10, fit_intercept=True, verbose=False):
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
            print(self.theta)
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold

def main():
	data = pd.read_excel('data3.xlsx')
	var1 = data['row1']
	var2 = data['row2']
	var3 = data['row3']
	var4 = data['row4']
	var5 = data['row5']
	x = []
	y = var5
	for i in data.index:
		temp = []
		# temp.append(1)
		temp.append(var1[i])
		temp.append(var2[i])
		temp.append(var3[i])
		temp.append(var4[i])
		x.append(temp)
	x = np.array(x)
	y = np.array(y)		
	xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.6)
	model = LogisticRegression()
	model.fit(xtrain, ytrain)
	preds = model.predict(xtest, 0.5)
	# print(preds)
	acc = (preds == ytest).mean()
	print(acc)
	# alpha=0.001; #learning rate
	# w = [0, 0, 0, 0, 0]
	# xtrain = np.array(xtrain)
	# ytrain = np.array(ytrain)
	# xtest = np.array(xtest)
	# ytest = np.array(ytest)
	# w = np.array(w)
	# xtest = np.array(xtest)
	# wt = w.transpose()
	# for epoch in range(100000):
		
	# 	for j in range(len(x[0])):
	# 		sumx = 0
	# 		for i in range(len(x)):
	# 			hx = sigmoid(np.dot(wt, x[i]))
	# 			val = ( y[i]-hx ) * x[i][j] 
	# 			sumx += val
	# 		# print(sumx)
	# 		wt[j] += alpha*sumx
	# 	print(w) 
if __name__ == "__main__":
	main()