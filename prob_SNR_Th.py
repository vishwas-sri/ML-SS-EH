# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 10:39:13 2023

@author: vishwas
This function calculate the prob of SNR>Th
"""

import numpy as np

# Given values
k = [2,3,5,9]
d = np.arange(1,21,1)
d = list(d)

P_SNR = np.zeros((len(k),len(d)))
P_SINR = np.zeros((len(k),len(d)))
Es = 0.05
# E = 0.3  # Energy of transmitted signal
g = 10**(-5)  # Path loss
tau = 100.0  # Duration of transmission time
sigma2 = 1e-07  # Power of AWGN
pw = 1e-06
mu = 0.02
# P_SNR = []
# P_SINR = []

for j in range(len(k)):
    for a in range(len(d)):
        E = k[j]*d[a]*Es
        # Generate a random sample of ζ from the exponential distribution with mean 1
        # zeta = np.random.exponential(scale=1)
    
        # Calculate SNR using the equation
        # SNR.append((E * g * zeta) / ((1 - mu) * tau * sigma2))
        # SINR.append((E * g * zeta) / ((1 - mu) * tau * (sigma2+pw)))
        P_SNR[j,a] = np.exp(-(((1 - mu) * tau * sigma2)*1)/(E * g ))
        P_SINR[j,a]= np.exp(-(((1 - mu) * tau * (sigma2+pw))*1)/(E * g ))

np.save('P_SNR', P_SNR)
np.save('P_SINR', P_SINR)
# print("SNR:", SNR)

 