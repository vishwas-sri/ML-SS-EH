#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 18:41:48 2023

@author: vishwas
"""
import numpy as np

# Simulation Parameters
T = 100e-3  # Time span for one slot 100ms
mu = 0.02   # Sensing duration ratio
t = mu*T    # Sensing time
fs = 50e3
PU = 1       # No. of PU
SU = 3       # No. of SU
Pr = 0.5    # Probability of spectrum occupancy
# Pd = 0.9    # Probability of detection
# Pf = 0.1    # Probability of false alarm
m = np.full(SU, 20)      # Battery capacity
Eh = 0.1    # Harvested energy during one slot
# Pw = -60    # Primary signal power in dBm
# PowerTx = 10**(Pw/10)  # Transmitted power
# Nw = -70    # Noise power in dBm
# PowerNo = 10**(Nw/10)
# g = 10**-5  # Path loss coefficeint 10^(-5)
# d = 500 # PU-SU distance in meters
samples = int(2*t*fs)
# samples = sample.astype(int) # No. of sample per sensing time
# w = 5e6     # Bandwidth
# samples = 50  # No. of sample
# N = SU
realize = 500
realize_test = 50000

Pd = []
Pf = []

 