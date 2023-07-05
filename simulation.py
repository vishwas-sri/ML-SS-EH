import numpy as np
variance = 2
T = 100  # Time span for one slot 100ms
mu = 0.02   # Sensing duration ratio
t = mu*T     # Sensing time
Pr = 0.5    # Probability of spectrum occupancy
Pw = -60    # Primary signal power in dBm
PowerTx = 10**(Pw/10)  # Transmitted power
Nw = -70    # Noise power in dBm
PowerNo = 10**(Nw/10)
g = 10**-5  # Path loss coefficeint 10^(-5)
d = 500  # PU-SU distance in meters


def MCS(realize, samples, SU):
    Y = np.zeros((realize, SU))
    S = np.zeros((realize))

    noisePower = PowerNo*np.ones(SU)

    for k in range(realize):
        n = gaussianNoise(noisePower, samples)
        H = channel(SU, d, g, variance, samples)
        X, S[k] = PUtx(samples, PowerTx, Pr, SU)
        PU = np.multiply(H.T, X)
        Z = PU + n
        Y[k, :] = np.sum(np.abs(Z)**2, axis=1)/(noisePower[0]*samples)
    return Y, S


def PUtx(samples, TXPower, Pr, N):
    S = 0
    X = np.zeros(samples)
    if (np.random.rand(1) <= Pr):
        S = 1
        X = np.random.randn(samples) * np.sqrt(TXPower)
    X = np.vstack([X]*N)
    return [X, S]


def gaussianNoise(noisePower, samples):
    N = len(noisePower)
    n = np.random.randn(N, samples) * np.sqrt(noisePower[0])
    return n


def channel(N, d, g, variance, samples):
    H = np.zeros(N)
    H = np.sqrt(-2 * variance * np.log(np.random.rand(N)))/np.sqrt(2)
    H = np.array(H*np.sqrt(d*(g)))  # Fading + path-loss (amplitude loss)
    H = np.vstack([H]*samples)
    return H
