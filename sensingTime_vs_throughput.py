# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:55:26 2023

@author: vishwas
this program plot the grpah between 
Throughput and sensing time
"""

import numpy as np
import matplotlib.pyplot as plt

# data = np.load('PI.npy')
# k = [2,3,5,9]
# d = [5,10,15,20]

# l = data[0,:]

Pd = np.load('Pd.npy')
Pf = np.load('Pf.npy')

T = 700
t = [10,30,50,100,150,200,300,350,400]
# t = np.arange(10,46,5)
ph0 = 0.5
ph1  = 0.5

# pd = np.array((0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.89))#,0.89,0.89,0.89))
# pf = np.array((0.8,0.7,0.63,0.52,0.42,0.35,0.25,0.21,0.19))
# pd1 = np.array((0.89,0.89,0.9,0.9,0.9,0.89,0.89,0.89,0.89))#,0.89,0.89,0.89))
# pf1 = np.array((0.81,0.73,0.66,0.55,0.46,0.39,0.28,0.26,0.22))
# pd = 0.9
# pf = 0.1

th = np.zeros((9,8))
th1 = []

for i in range(9):
    for j in range(8):
        # R1 = (T-t[i])/T*(ph0*(1-pf[i]))* 6.6582
        # R2 = (T-t[i])/T*(ph1*(1-pd[i]))* 6.6137
        R1 = (T-t[i])/T*(ph0*(1-Pf[i,j]))
        R2 = (T-t[i])/T*(ph1*(1-Pd[i,j]))
        th[i,j]=R1+R2
        # R11 = (T-t[i])/T*(ph0*(1-pf1[i]))
        # R22 = (T-t[i])/T*(ph1*(1-pd1[i]))
        # th1.append(R11+R22)
        # th.append(l[i]*(ph0*(1-pf)+ph1*(1-pd)))
time = np.array(t)
time = time/(2*500)
label = ['Linear SVM','Gaussian SVM', 'Logistic','NBC','S1','OR','AND','MRC']
marker = ['*','3','1','d', 'v','^','2','o']
for i in range(8):
    plt.plot(time,th[:,i], label = label[i], marker = marker[i])
# plt.plot(time,th1,'-bo', label = 'OR')
plt.xlabel('Sensing time in ms')
plt.ylabel('Throughput in bps')
# plt.xtics(np.arrange(0,0.5,0.05))
plt.title(' ')
plt.legend(loc='lower right')
plt.grid()
plt.show()
