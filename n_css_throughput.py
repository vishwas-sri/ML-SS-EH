# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 15:46:59 2023

@author: vishwas
This script plot the average secondary
throughput for N-CSS
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.load('PI.npy')
k = [2,3,5,9]
d = [5,10,15,20]

P_SNR = np.load('P_SNR.npy')
P_SINR = np.load('P_SINR.npy')

ph0 = 0.5
ph1  = 0.5
pd = 0.9
pf = 0.1

for i in range(len(data)):
    marker = ['s','o','D','^']
    l = data[i,:]
    l = (l*(ph0*(1-pf)*P_SNR[i,:] +ph1*(1-pd)*P_SINR[i,:]))
    plt.plot(l, marker = marker[i],label = 'k = %d'%(k[i]))
plt.xticks(np.linspace(0, 20,5))
plt.xlabel('Energy threshold $\delta$')
plt.ylabel('Average SU throughput')
plt.legend()
plt.grid()
plt.show()
# l2 = data[1,:]
# l2 = (l2*(ph0*(1-pf)*P_SNR[1,:] +ph1*(1-pd)*P_SINR[1,:]))
# plt.plot(l2)