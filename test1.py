# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:55:26 2023

@author: vishwas
"""

import numpy as np
import matplotlib.pyplot as plt

T = 1000
t = [10,30,50,100,150,200,250,300]
# t = np.arange(10,46,5)
ph0 = 0.5
ph1  = 0.5

pd = np.array((0.2786,0.277,0.310,0.3488,0.34418,0.36704,0.36488,0.36328))
pf = np.array((0.18472,0.1159,0.114,0.0882,0.0506,0.04898,0.0322,0.0227))
# pd = 0.9
# pf = 0.1

th = []

for i in range(5):
    # R1 = (T-t[i])/T*(ph0*(1-pf))* 6.6582
    # R2 = (T-t[i])/T*(ph1*(1-pd))* 6.6137
    R1 = (T-t[i])/T*(ph0*(1-pf[i]))
    R2 = (T-t[i])/T*(ph1*(1-pd[i]))
    th.append(R1+R2)

plt.plot(th)
plt.grid()
plt.show()
