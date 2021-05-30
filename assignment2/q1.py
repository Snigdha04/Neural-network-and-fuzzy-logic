# hebbian learning
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlrd
from pandas import DataFrame

def hebbLearn1(x, y, nf):
    if nf==0 :
        w = [0, 0];
    else:
        w = 0;
    b = 0;
    xt = np.transpose(x);
    alpha = 0.1;
    theta = 0.5;
    iterations = 10;
    cost = [];
    for t in range(iterations):
        sume = 0.0;
        # print(w);
        for m in range(np.size(x[0])):
            if nf==0 :
                am = b + np.matmul(xt[m], w);
            else:
                am = b + xt[m]*w;
            hm = 1 if am >= theta else 0;
            if hm != y[m] :
                w = w + alpha*y[m]*xt[m];
                b = b + alpha*y[m];
            sume += (y[m]-hm)**2;
        # print(sume);
        cost.append(sume);
    plt.plot(range(iterations), cost);
    plt.show();

def main():
    # AND gate
    x = [[0, 0, 1, 1],[0, 1, 0, 1]];
    x = np.array(x);
    y = [0, 0, 0, 1];
    y = np.array(y);
    hebbLearn1(x, y, 0);

    # OR gate
    x = [[0, 0, 1, 1],[0, 1, 0, 1]];
    x = np.array(x);
    y = [0, 1, 1, 1];
    y = np.array(y);
    hebbLearn1(x, y, 0);

    # NOT gate
    x = [0, 1];
    x = np.array(x);
    y = [1, 0];
    y = np.array(y);
    hebbLearn1(x, y, 1);

    # XOR gate
    x = [[0, 0, 1, 1],[0, 1, 0, 1]];
    x = np.array(x);
    y = [0, 1, 1, 0];
    y = np.array(y);
    hebbLearn1(x, y, 0);

    # ANDNOT gate
    x = [[0, 0, 1, 1],[0, 1, 0, 1]];
    x = np.array(x);
    y = [0, 0, 1, 0];
    y = np.array(y);
    hebbLearn1(x, y, 0);

    # NAND gate
    x = [[0, 0, 1, 1],[0, 1, 0, 1]];
    x = np.array(x);
    y = [1, 1, 1, 0];
    y = np.array(y);
    hebbLearn1(x, y, 0);

    #  NOR gate
    x = [[0, 0, 1, 1],[0, 1, 0, 1]];
    x = np.array(x);
    y = [1, 0, 0, 0];
    y = np.array(y);
    hebbLearn1(x, y, 0);

    #  XNOR gate
    x = [[0, 0, 1, 1],[0, 1, 0, 1]];
    x = np.array(x);
    y = [1, 0, 0, 1];
    y = np.array(y);
    hebbLearn1(x, y, 0);

if __name__ == "__main__":
	main()
