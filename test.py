import numpy as np
from scipy.stats import rayleigh

T = 100       # Time span for one slot (100ms)
mu = 0.02     # Sensing duration ratio
t = mu * T    # Sensing time
PU = 1        # No. of PUs
SU = 3        # No. of SUs
Pr = 0.5      # Probability of spectrum occupancy
Pw = -60      # Primary signal power in dBm
Nw = -70      # Noise power in dBm
g = 10**-5    # Path loss coefficient 10^(-5)
training_samples = 250  # No. of samples
testing_samples = 5000

X_train = np.zeros((training_samples, SU))   # Array to store received signal energy at SUs
y_train = np.zeros((training_samples,))      # Array to store PU availability

X_test = np.zeros((testing_samples, SU))   # Array to store received signal energy at SUs
y_test = np.zeros((testing_samples,))      # Array to store PU availability

def dataset_gen(samples):
    dataset_X = np.zeros((samples, SU))   # Array to store received signal energy at SUs
    dataset_y = np.zeros((samples,))
    for i in range(samples):
    # Generate random PU availability (0 or 1)
        dataset_y[i] = np.random.choice([0, 1], p=[1 - Pr, Pr])
    
        if dataset_y[i] == 1:
        # Calculate received signal energy at SUs for available PU
            for j in range(SU):
                # Generate Rayleigh-distributed channel gain
                h = rayleigh.rvs(scale=np.sqrt(g/2))
                
                # Calculate received signal energy
                received_signal_energy = 10**(0.1 * (Pw + 10 * np.log10(h)))
                
                # Add AWGN noise
                noise_power = 10**(0.1 * Nw)
                received_signal_energy += np.random.normal(0, np.sqrt(noise_power))
                
                dataset_X[i, j] = received_signal_energy
    return dataset_X, dataset_y

X_train, y_train = dataset_gen(samples=training_samples)
X_test, y_test = dataset_gen(samples=testing_samples)

print(X_train)