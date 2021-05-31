# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r"C:\Users\dell\Desktop\code\knn")
import get_data

def EuclideanDistances(x):
    """

    Parameters
    ----------
    x : ndarray
        matrix of instances.
    norm : float, optional
        p-normal for distances. The default is 2.

    Returns
    -------
    distances : ndarray
        matrix of distances between instances.

    """
    n, p = x.shape
    xy = x @ x.T
    xsq = np.sum(np.square(x), axis=1).reshape(n,1)
    distances = np.sqrt(xsq + xsq.T - 2*xy + np.eye(n)*np.exp(10))
    
    return distances
    
    
def Knn(x, y, k=5):
    """

    Parameters
    ----------
    x : ndarray
        matrix of instances.
    y : array
        labels of instances.
    k : int, optional
        the number of neighbours used to assign instances' label. The default is 10.

    Returns
    -------
    pre : array
        predict labels.

    """
    n, p = x.shape
    pre = np.zeros((n,1))
    distances = EuclideanDistances(x)
    kth_value = np.sort(distances,axis=1)[:,k]
    for i in range(n):
        position = np.where(distances[i,:]<=kth_value[i])[0]
        labels = y[position].astype("int32")
        pre[i] = np.argmax(np.bincount(labels))
    
    return pre
    
def Score(y, pre):
    """

    Parameters
    ----------
    y : array
        real labels.
    pre : array
        predict labels.

    Returns
    -------
    score : float
        model score.

    """
    n = y.shape[0]
    pre = pre.reshape(n,)
    score = np.sum(y==pre)/n
    
    return score


n = 10000
pos = np.array([0.1, 0.2, 0.3, 0.4])
data = get_data.FourCluster(n, pos)
x = data[:,:2]
y = data[:,-1]

pre = Knn(x, y, k=5)
score = Score(y, pre)

plt.figure()
plt.scatter(data[:,0], data[:,1], c=y)
plt.show()

plt.figure()
plt.scatter(data[:,0], data[:,1], c=pre)
plt.show()
    
        











