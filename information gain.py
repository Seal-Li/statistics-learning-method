# -*- coding: utf-8 -*-
import numpy as np


def EmpiricalEntropy(y):
    """

    Parameters
    ----------
    y : int
        real laabels of instances.

    Returns
    -------
    H : float
        emperical entropy.

    """
    n = y.shape[0]
    ylabels = list(set(y))
    nc = len(ylabels)
    pos_y = np.zeros((nc,1))
    
    for i in range(nc):
        pos_y[i] = np.sum(y==ylabels[i])/n
        
    H = -(pos_y.T @ np.log2(pos_y))
    
    return H


def EmpiricalConditionEntropy(A, y):
    """

    Parameters
    ----------
    A : int
        A feature of data, it's a categorical variables.
    y : int
        real laabels of instances.

    Returns
    -------
    Hc : float
        empirical condition entropy.

    """
    n = y.shape[0]
    Alabels = list(set(A))
    nA = len(Alabels)
    ylabels = list(set(y))
    nc = len(ylabels)
    
    Pik = np.zeros((nA,nc))  # Dik的概率
    Hc = 0
    for i in range(nA):
        subA = A[A==Alabels[i]]
        ni = len(subA)
        position_subA = np.where(A==Alabels[i])[0]
        for k in range(nc):
            Pik[i,k] = np.sum(y[position_subA]==ylabels[k])/ni
            if Pik[i,k]==0:
                continue
            Hc = Hc - (ni/n)*(Pik[i,k]*np.log2(Pik[i,k]))
            
    return Hc


def InformationGain(x, y, criterion="information gain ratio"):
    """

    Parameters
    ----------
    x : ndarray
        matrix of instances.
    y : int
        real labels of instances.
    criterion : str
        method of evaluation for feature selection

    Returns
    -------
    BestFeature : int
        index of the best feature.

    """    
    p = x.shape[1]
    H = EmpiricalEntropy(y)
    info_gain = np.zeros((p,1))
    info_gain_ratio = np.zeros((p,1))
    
    for i in range(p):
        A = x[:,i]
        HA = EmpiricalEntropy(A)
        Hc = EmpiricalConditionEntropy(A, y)
        info_gain[i] = H - Hc
        info_gain_ratio[i] = info_gain[i]/HA
        
    if criterion=="information gain ratio":
        BestFeature = np.argmax(info_gain_ratio) + 1
    else:
        BestFeature = np.argmax(info_gain) + 1
    print("the best feature is: A{}".format(BestFeature))
    
    return BestFeature


path = r"C:\Users\dell\Desktop\code\decision tree\data.txt"
data = np.genfromtxt(path,delimiter=",", skip_header=1)
x = data[:,1:-1]
y = data[:,-1]
BestFeature = InformationGain(x, y, criterion="info_gain_ratio")


    




















