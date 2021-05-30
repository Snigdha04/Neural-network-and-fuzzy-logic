# MFNN 4 LAYERED
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlrd
from pandas import DataFrame
from numpy.linalg import inv
import math
from sklearn.cross_validation import train_test_split
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
    # print(sigmoid(xtra))
    nhid1 = 8
    nhid2 = 6
    nout = 3
    # print(len(xtra[0]))
    yt = []
    for i in range(len(ytr)):
        if ytr[i] == 1: yt.append([1, 0, 0])
        elif ytr[i] == 2: yt.append([0, 1, 0])
        else: yt.append([0, 0, 1])

    wl1 = np.zeros((nhid1, len(xtra[0])))
    wl2 = np.zeros((nhid2, nhid1))
    wl3 = np.zeros((nout, nhid2))
    b1 = np.zeros(nhid1)
    b2 = np.zeros(nhid2)
    b3 = np.zeros(nout)
    yt = np.array(yt)
    eta = 0.1
    iteration = 1000
    cost = np.zeros(iteration)
    cost_test = np.zeros(iteration)
    for ite in range(iteration):
        for m in range(len(xtra)):
            out0 = xtra[m]
            out1 = sigmoid(np.matmul(out0, wl1.T)+b1)
            out2 = sigmoid(np.matmul(out1, wl2.T)+b2)
            out3 = sigmoid(np.matmul(out2, wl3.T)+b3)

            delta3 = (yt[m]-out3)*out3*(1-out3)
            delta2 = np.matmul(delta3, wl3)*out2*(1-out2)
            delta1 = np.matmul(delta2, wl2)*out1*(1-out1)

            wl3 = wl3 + eta*np.outer(delta3, out2)
            b3 = b3 + eta*delta3
            wl2 = wl2 + eta*np.outer(delta2, out1)
            b2 = b2 + eta*delta2
            wl1 = wl1 + eta*np.outer(delta1, out0)
            b1 = b1 + eta*delta1
            cost[ite] += np.sum((yt[m]-out3)**2)

        # print(wl1)
        # print(wl2)
        # testing the data
        # accuracy is the objective function
        cnt = 0
        tot = 0
        for m in range(len(xte)):
            out0 = xte[m]
            out1 = sigmoid(np.matmul(out0, wl1.T)+b1)
            out2 = sigmoid(np.matmul(out1, wl2.T)+b2)
            out3 = sigmoid(np.matmul(out2, wl3.T)+b3)
            cost_test[ite] += np.sum((yte[m]-out3)**2)
            ind = find_max(out3)+1
            if ind==yte[m] : cnt+=1
            tot+=1

    cost /= len(xtra)
    cost_test /= len(xte)
    plt.plot(range(iteration), cost)
    # plt.plot(range(iteration), cost_test)
    plt.show()

    print('Accuracy : ')
    print(cnt/tot)

if __name__ == "__main__":
	main()
