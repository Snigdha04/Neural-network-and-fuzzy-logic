# clustering with k=2
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import xlrd
from pandas import DataFrame
from numpy.linalg import inv
def dist(a, b):
	sm=0
	for i in range(len(a)):
		sm += (a[i]-b[i])**2
	return sm

data = pd.read_excel('data2.xlsx')
# print(data)
var1 = data['row1']
var2 = data['row2']
var3 = data['row3']
var4 = data['row4']
v1 = [1, 1, 1, 1]
v2 = [2, 10, 5, 2]
x = []
for i in data.index:
	temp = []
	temp.append(var1[i])
	temp.append(var2[i])
	temp.append(var3[i])
	temp.append(var4[i])
	x.append(temp)
for epoch in range(10):
	cls1 = []
	cls2 = []
	for i in range(len(x)):
		if dist(x[i], v1)<dist(x[i], v2):
			cls1.append(x[i])
		else:
			cls2.append(x[i])
	# find new centre
	v1 = [0, 0, 0, 0]
	for i in range(len(cls1)):
		for j in range(len(cls1[0])):
			v1[j] += cls1[i][j]
	v1 = np.array(v1)
	if len(cls1)!= 0:
		v1 = v1/len(cls1)

	v2 = [0, 0, 0, 0]
	for i in range(len(cls2)):
		for j in range(len(cls2[0])):
			v2[j] += cls2[i][j]
	v2 = np.array(v2)
	if len(cls2)!= 0:
		v2 = v2/len(cls2) 
	print(v1)
	print(v2)