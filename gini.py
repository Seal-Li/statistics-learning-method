# -*- coding: utf-8 -*-
import numpy as np

def Gini(y):
    """

    Parameters
    ----------
    y : array
        real labels of instances.

    Returns
    -------
    gini : float
        index of gini.

    """
    n = y.shape[0]
    ylabels = list(set(y))
    nc = len(ylabels)
    pos_y = np.zeros((nc,1))
    
    for i in range(nc):
        pos_y[i] = np.sum(y==ylabels[i])/n
    gini = 1 - np.sum(np.square(pos_y))
    
    return gini


def ConditionGini(A, y):
    """

    Parameters
    ----------
    A : array
        a feature of instances.
    y : array
        real labels of instances.

    Returns
    -------
    condition_gini : float
        the index of gini under the condition feature A was given.

    """
    n = y.shape[0]
    Alabels = list(set(A))
    nc = len(Alabels)
    condition_gini = np.zeros((nc,))
    
    for i in range(nc):
        p = np.sum(A==Alabels[i])/n
        sub_y1 = y[A==Alabels[i]]
        sub_y2 = y[A!=Alabels[i]]
        condition_gini[i] += p*Gini(sub_y1) + (1-p)*Gini(sub_y2)
    split_point = np.argmax(condition_gini)
    
    print("the best split point is the {}th value".format(split_point+1))
    
    return condition_gini

path = r"C:\Users\dell\Desktop\code\decision tree\data.txt"
data = np.genfromtxt(path,delimiter=",", skip_header=1)
x = data[:,1:-1]
y = data[:,-1]
n, p = x.shape
gini = Gini(y)

for i in range(p):
    A = x[:,i]
    condition_gini = ConditionGini(A, y)
























