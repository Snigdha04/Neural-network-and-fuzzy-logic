# vectorised linear regression
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import xlrd
from pandas import DataFrame
from numpy.linalg import inv

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
matx = np.array(x)
maty = np.array(y)
matxt = matx.transpose()
wght = np.matmul(matxt, matx)
wght = inv(wght)
wght = np.matmul(wght, matxt)
wght = np.matmul(wght, y)
print(wght)