"""
This is the main file that perform
spectrum sensing
"""
import copy
import numpy as np
import simulation as sm
from model import Classification
import matplotlib.pyplot as plt
import plot


# Simulation Parameters
T = 100e-3  # Time span for one slot 100ms
mu = 0.02   # Sensing duration ratio
t = (np.arange(0.1,2.1,0.1))*1e-3 #mu*T    # Sensing time
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
sample = (2*t*fs)
sample = sample.astype(int) # No. of sample per sensing time
# w = 5e6     # Bandwidth
# samples = 50  # No. of sample
# N = SU
realize = 500
realize_test = 50000

th=[]

for j in range(len(sample)):
    samples = sample[j]
    # MCS(realize,samples,SU)
    X_train, y_train, _ = sm.MCS(realize, samples, SU)
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
    # file.append(demo.Gaussian_SVM())
    # file.append(demo.Logistic())
    # file.append(demo.NaiveBayes())
    # file.append(demo.S1())
    # file.append(demo.OR())
    # file.append(demo.AND())
    # file.append(demo.MRC())
    # Pf,Pd,_,_,_ = file[0]
    
    # idx = np.where(Pf>=0.2)[0]
    
    # plt.plot(Pf,Pd)
    # plt.grid(True)
    # plt.show()
    _,_,_,_,y_p = file[0]
    
    R = 0
    W = 0
    
    total = len(y_test)
    
    PH1 = (np.sum(y_test)/total)
    PH0 = 1-PH1
    
    for y_a, y_p in zip(y_test, y_p):
        if y_a == 1 and y_p == 0:
            W += 1
        if y_a == 0 and y_p == 0:
            R += 1
    
    one_pf = R/total
    one_pd = W/total
    
    th1 = (T-t[j])/T
    th.append(th1*(PH1*one_pd+PH0*one_pf))
    # th.append(th1*(PH0*one_pf))
    # print(th)
    
    
    # if file:
    #     plot.show_plot(file)  # , mark
    # plt.show()
    print('Sample ',j,'completed')
    
plt.plot(sample,th)
plt.grid(True)
plt.show()
