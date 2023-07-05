import numpy as np
import simulation as sm
import model 
import plot
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
samples = 500  # No. of sample per sensing time
# w = 5e6     # Bandwidth
# samples = 50  # No. of sample
# N = SU
realize = 250
realize_test = 50000

# MCS(realize,samples,SU)
X_train, y_train = sm.MCS(realize, samples, SU)
X_test, y_test = sm.MCS(realize_test, samples, SU)

fpr, tpr, auc = model.ml_model(X_train, y_train, X_test, y_test) #, mark

plot.show_plot(fpr, tpr, auc) #, mark
