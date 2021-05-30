#  likelihood ratio test (LRT)
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import xlrd
from pandas import DataFrame
from numpy.linalg import inv
import math
from sklearn.cross_validation import train_test_split
from numpy.linalg import inv
from numpy import cov

def main():
	data = pd.read_excel('data3.xlsx')
	var1 = data['row1']
	var2 = data['row2']
	var3 = data['row3']
	var4 = data['row4']
	var5 = data['row5']
	x = []
	for i in data.index:
		temp = []
		# temp.append(1)
		temp.append(var1[i])
		temp.append(var2[i])
		temp.append(var3[i])
		temp.append(var4[i])
		temp.append(var5[i])
		x.append(temp)
	x = np.array(x)	
	x1 = []
	x2 = []
	# print(x)
	# print('printed x\n')
	xtrain, xtest, ytrain, ytest = train_test_split(x[:,:4], x[:,4], train_size=0.6)
	for i in range(len(ytrain)):
		if ytrain[i] == 0:
			temp = xtrain[i].tolist()
			# temp.append(1)
			x1.append(temp)
		else:
			temp = xtrain[i].tolist()
			x2.append(temp)
	# print(x1)
	# finding the means
	u1 = [0,0,0,0]
	u1 = np.array(u1)
	x1 = np.array(x1)
	for xx in x1:
		u1 = np.add(u1, xx)
	u1 = u1/len(x1)

	u2 = [0,0,0,0]
	u2 = np.array(u2)
	x2 = np.array(x2)
	for xx in x2:
		u2 = np.add(u2, xx)
	u2 = u2/len(x2)
	cov1 = cov(x[:,:4].T)
	cov1 = inv(cov1)
	print(cov1)
	# [[ 11.74272179  -7.24443147  -7.60918297   5.13458222]
	#  [ -7.24443147  11.4468994    7.29671775  -6.39827837]
	#  [ -7.60918297   7.29671775  17.29107951 -33.05113455]
	#  [  5.13458222  -6.39827837 -33.05113455  78.26853974]]

	w = np.dot(cov1, (u2 - u1))
	print(w)
	# [-0.09231329  0.91722705 -1.33392136  0.56030023]

	b = (1/2)*(np.dot(u2.T, np.dot(cov1, u2)) - np.dot(u1.T, np.dot(cov1, u1)))
	print(b)
	# -1.06923186932

	# computing the accuracy
	y = []
	for xx in xtest:
		temp = np.dot(w.T, xx) + b
		if temp<0:
			y.append(0)
		else:
			y.append(1)
	y = np.array(y)
	acc = (y == ytest).mean()
	print(acc)
	# 0.65

if __name__ == "__main__":
	main()