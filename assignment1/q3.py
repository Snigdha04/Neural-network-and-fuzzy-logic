# linear regression algorithm
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import xlrd
from pandas import DataFrame

data = pd.read_excel('data.xlsx')
# print(data)
var1 = data['row1']
var2 = data['row2']
y = data['row3']
x = []
for i in data.index:
	temp = []
	temp.append(1);
	temp.append(var1[i])
	temp.append(var2[i])
	x.append(temp)
# print(x)
alpha=0.000000001; #learning rate
lmd = 1
w = [1,1,1]
for epoch in range(10):
	for j in range(len(x[0])):
		# print("me");
		sumx = 0
		for i in range(len(x)):
			hx = w[0] + w[1]*x[i][1] + w[2]*x[i][2]
			hx = hx - y[i]
			hx = hx*x[i][j]
			sumx += hx
			# print(hx)
		# print(hx)
		w[j] = (1-alpha*lmd)*w[j] - alpha*sumx
	print(w)
	print("\n")
