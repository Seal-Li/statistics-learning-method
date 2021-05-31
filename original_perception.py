# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def func(x, y, w, b):
    p = w.shape[0]
    f = 0
    for i in range(p):
        f = f + x[i]*w[i]
    return y*(f+b)


def epoch(x, y, w, b, lr = 1):
    """

    Parameters
    ----------
    x : ndarray
        data matrix.
    y : int
        data labels.
    w : ndarray
        weight vector.
    b : float
        bias.
    lr : float, optional
        learning rate. The default is 1.

    Returns
    -------
    flag : int
        number of points were not correctly assigned.

    """
    flag = 0
    n, p = x.shape
    
    for i in range(n):
        if func(x[i,:], y[i], w, b)<=0:
            # print("Point {} error, value{}".format(i, func(x[i,:], y[i], w, b)))
            flag = flag + 1
            for j in range(p):
                w[j,0] = w[j,0] + lr*y[i]*x[i,j]
            b = b + lr*y[i]
            
    return w, b, flag


path = r"C:\Users\dell\Desktop\code\perception\double_bands.txt"
data = np.genfromtxt(path)
x = data[:,:2]
y = data[:,-1]
n, p = x.shape
w = np.zeros((p,1)) + np.array([[-100],[-5000]])
b = -1000

flag = 1
num = 1
while flag:
    w, b, flag = epoch(x, y, w, b)
    print("epoch:{},error points:{}".format(num, flag))
    num = num + 1

# 防止精度溢出
eps = np.exp(-22)
lower_limit = np.exp(-1)
if w[1,0]<=eps:
    w[1,0] = lower_limit
l1 = np.arange(np.min(x[:,0]),np.max(x[:,0]))
l2 = -(w[0,0]*l1 + b)/w[1,0]  # separating Hyperplane

fig = plt.figure()
plt.scatter(x[:,0], x[:,1], c=y, s=(10*72./fig.dpi)**2, marker="s")
plt.plot(l1, l2, c="blue")
plt.show()

            

