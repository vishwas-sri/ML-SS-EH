import numpy as np
import simulation as sm
from model import Classification
import plot
import matplotlib.pyplot as plt
import copy
# Simulation Parameters
# T = 100  # Time span for one slot 100ms
# mu = 0.02   # Sensing duration ratio
# t = mu*T    # Sensing time
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
samples = 50  # No. of sample per sensing time
# w = 5e6     # Bandwidth
# samples = 50  # No. of sample
# N = SU
realize = 250
realize_test = 50000

# MCS(realize,samples,SU)
X_train, y_train, SNR = sm.MCS(realize, samples, SU)
X_test, y_test, SNR = sm.MCS(realize_test, samples, SU)

# SNR2 = []
X_test_2 = copy.deepcopy(X_test)
for i in range(SU):
    # SNR2.append(SNR[0][i])
    # SNR = SNR2
    NormSNR = [x/np.sum(SNR) for x in SNR]
for i in range(SU):
    X_test_2[:,i] = X_test_2[:,i]*NormSNR[i]

file = []

demo =Classification(X_train=X_train,y_train=y_train,X_test=X_test,
                    y_test=y_test, samples=samples,SU=SU, X_test_2=X_test_2)


file.append(demo.Linear_SVM())
file.append(demo.Gaussian_SVM())
file.append(demo.S1())
file.append(demo.OR())
file.append(demo.AND())
file.append(demo.MRC())

if (file != []):
    plot.show_plot(file)  # , mark
# plt.show()
