#  RBFNN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlrd
from pandas import DataFrame
from numpy.linalg import pinv
import math
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans

def find_max(x):
    index = 0
    maxi = x[index]
    for i in range(len(x)):
        if x[i]>maxi:
            maxi = x[i]
            index = i
    return index

def sigmoid(x):
    temp = 1.0 + np.exp(-x)
    return (1.0/temp)

def basis_func(x, mu):
    # taking kernel function to be cubic function
    val = (np.linalg.norm(x-mu))**3
    return val

def main():
    data = pd.read_excel('dataset.xlsx')
    var1 = data['row1']
    var2 = data['row2']
    var3 = data['row3']
    var4 = data['row4']
    var5 = data['row5']
    var6 = data['row6']
    var7 = data['row7']
    y = data['row8']
    x = []
    for i in data.index:
        temp = []
        # temp.append(1)
        temp.append(var1[i])
        temp.append(var2[i])
        temp.append(var3[i])
        temp.append(var4[i])
        temp.append(var5[i])
        temp.append(var6[i])
        temp.append(var7[i])
        x.append(temp)

    x = np.array(x)
    y = np.array(y)
    # x = sigmoid(x)
    # normalizing the data
    mean = np.sum(x,axis = 0)/len(x)
    variance = (np.sum((x - mean)**2,axis = 0))/len(x)
    x = (x - mean)/variance

    xtra, xte, ytr, yte = train_test_split(x, y, train_size=0.7)

    yt = []
    for i in range(len(ytr)):
        if ytr[i] == 1: yt.append([1, 0, 0])
        elif ytr[i] == 2: yt.append([0, 1, 0])
        else: yt.append([0, 0, 1])

    nhid = 8
    kmeans = KMeans(n_clusters=nhid, random_state=0).fit(xtra)
    mu = kmeans.cluster_centers_

    # training the data
    h = np.zeros((len(xtra), nhid))
    for i in range(len(xtra)):
        for j in range(nhid):
            h[i][j] = basis_func(xtra[i], mu[j])

    w = np.matmul(pinv(h), yt)

    # testing the data
    # accuracy is the objective function
    cnt = 0
    tot = 0
    ht = np.zeros((len(xte), nhid))
    for i in range(len(xte)):
        for j in range(nhid):
            ht[i][j] = basis_func(xte[i], mu[j])
    yp = np.matmul(ht, w)
    for i in range(len(yp)):
        ind = find_max(yp[i])+1
        if ind==yte[i] : cnt+=1
        tot+=1

    print('Accuracy : ')
    print(cnt/tot)

if __name__ == "__main__":
	main()
