import numpy as np

# DEFAULT PARAMETER
Pr = 0.5                                                    # PU activity probability		
TXPower = 0.1                                               # PU transmission power
w = 5e6                                                     # Bandwidth
NoisePSD_dBm = -153                                         # Noise power spectral density	
# T=5e-6                                                      # sensing time

def MCS(realiz,T=0.5e-6,kind='ray',variance=2,SuNumber=3):
    PU = np.array([0,0])*1e3                                # PU position 	
    SU = np.dstack(( np.zeros((1,SuNumber)),                # SU position
                    np.linspace(0.5,1,SuNumber)
                    [np.newaxis,:] ))[0]*1e3
    if SuNumber==1:
        SU=np.array([[0 ,750]])
    N= SuNumber             
    noisePower= (10**(NoisePSD_dBm/10)*1e-3)*w*np.ones(N)   # Noise power
    a= 4                                                    # Path loss factor                            
    samples=round(2*T*w)                                    # No. of samples
    
    # PU-SU distance
    d= np.zeros(N)
    for i in range(N):
        d[i] = np.linalg.norm(PU-SU[i,:])
    
        
    Y = np.zeros((realiz,N))                 
    S = np.zeros(realiz)                   
    SNR = np.zeros((N,realiz))            
        
    for k in range(realiz):
        n = gaussianNoise(noisePower,samples)
        H = channel(N,d,a,kind,variance,samples)                               
        X, S[k] = PUtx(samples,TXPower, Pr, N)
        PU=np.multiply(H.T,X)
        Z = PU + n
        
        SNR[:,k] = np.mean(np.abs(PU)**2,axis=1)/noisePower[0]
        Y[k,:] = np.sum(np.abs(Z)**2,axis=1)/(noisePower[0]*samples)

    meanSNR = np.mean(SNR[:,S==1],1)
    meanSNRdB = 10*np.log10(meanSNR)
    
    return Y,S,meanSNRdB
    
def PUtx(samples,TXPower, Pr, N ):
    S = 0
    X = np.zeros(samples)
    if (np.random.rand(1) <= Pr):
        S=1
        X=np.random.randn(samples) * np.sqrt(TXPower)
    X= np.vstack([X]*N)
    return [X,S]

def gaussianNoise(noisePower,samples):
    N = len(noisePower)
    n= np.random.randn(N,samples)  * np.sqrt(noisePower[0])
   
    return n

def channel(N,d,a,kind,variance,samples):
    H=np.zeros(N)
    if (kind=='ray'):
        H=np.sqrt(-2 * variance * np.log(np.random.rand(N)))/np.sqrt(2)
    # elif kind=='nakagami':
    #     m=1.5
    #     omega=1
    #     H = np.sqrt(np.gamrnd(m,omega/m,[N,1]))*np.sqrt(variance)
    # elif kind=='rician':
    #     # for i in range(N)
    #     #     ricChan= comm.RicianChannel("MaximumDopplerShift",
          #      0,"NumSamples",1,"ChannelFiltering",0)
    #     #     H(i,:) = abs(ricChan())
    #     # end
    #     # H=H*sqrt(variance)
    #     pass
    else:
        H = np.ones(N,1)
    H = np.array(H*np.sqrt(d**(-a)))# Fading + path-loss (amplitude loss)
    H = np.vstack([H]*samples)
    

    return H
