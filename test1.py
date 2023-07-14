# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:55:26 2023

@author: vishwas
"""

import numpy as np
import matplotlib.pyplot as plt

T = 700
t = [10,30,50,100,150,200,300,350,400]
# t = np.arange(10,46,5)
ph0 = 0.5
ph1  = 0.5

pd = np.array((0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.89,0.89,0.89,0.89))
pf = np.array((0.8,0.7,0.63,0.52,0.42,0.35,0.25,0.21,0.19))
# pd = 0.9
# pf = 0.1

th = []

for i in range(9):
    # R1 = (T-t[i])/T*(ph0*(1-pf))* 6.6582
    # R2 = (T-t[i])/T*(ph1*(1-pd))* 6.6137
    R1 = (T-t[i])/T*(ph0*(1-pf[i]))
    R2 = (T-t[i])/T*(ph1*(1-pd[i]))
    th.append(R1+R2)

plt.plot(t,th,'-r*')
plt.grid()
plt.show()
