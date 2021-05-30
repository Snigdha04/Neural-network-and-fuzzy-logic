import sklearn
from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train =sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import xlrd
from pandas import DataFrame
from numpy.linalg import inv
import math

data = pd.read_excel('data3.xlsx')
# print(data)
var1 = data['row1']
var2 = data['row2']
var3 = data['row3']
var4 = data['row4']
var5 = data['row5']
x = []
y = []
xtest = []
ytest = []
# print(len(data))
for i in data.index:
	temp = []
	# temp.append(1)
	temp.append(var1[i])
	temp.append(var2[i])
	temp.append(var3[i])
	temp.append(var4[i])
	if i<len(data)*0.6:
		x.append(temp)
		y.append(var5[i])
	else:
		xtest.append(temp)
		ytest.append(var5[i])

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x =sc_X.fit_transform(x)
xtest = sc_X.transform(xtest)
#Fitting Logistic Regression to dataset
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
print(classifier.fit(x, y))
 
#Predicting the test set result
y_pred = classifier.predict(xtest)
 
#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, y_pred)
print(cm)
score = classifier.score(xtest, ytest)
print(score)