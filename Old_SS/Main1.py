from Algorithms1 import Classification
# import os
import pandas as pd
# from sklearn.model_selection import train_test_split
from pathlib import Path
base_path = Path(__file__).parent
# from scipy import special
import numpy as np
import matplotlib.pyplot as plt
# from sklearn import metrics
import copy

# def C():
file_path1 = (base_path/"../SS v01/Data/NAK/4_Train.csv").resolve()
dataset = pd.read_csv(file_path1,header=None)
X_train= dataset.iloc[:, 0:-1].values
y_train = dataset.iloc[:, -1].values
   
file_path2 = (base_path/"../SS v01/Data/NAK/4_Test.csv").resolve()
dataset = pd.read_csv(file_path2,header=None)
X_test = dataset.iloc[:, 0:-1].values
y_test = dataset.iloc[:, -1].values

file_path3 = (base_path/"../SS v01/Data/NAK/4_SNR.csv").resolve()
SNR= pd.read_csv(file_path3,header=None)   
Samples=50
# print("X_train",X_train)
# print("y_train",y_train)
# print("X_test",X_test)
# print("y_test",y_test)
# y = y.reshape(len(y),1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

SNR2=[]
SU=len(SNR)

X_test_2=copy.deepcopy(X_test)    
for i in range(SU):
    SNR2.append(SNR[0][i])
SNR=SNR2
NormSNR=[x/np.sum(SNR) for x in SNR]
for i in range(SU):
    X_test_2[:,i]=X_test_2[:,i]*NormSNR[i]

demo = Classification(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                    Samples=Samples, SU=SU, X_test_2=X_test_2, SNR=SNR)

# to execute any file, un-comment it
file=[]
file2=[]
file.append(demo.Logistic())
file.append(demo.MLP())
file.append(demo.NaiveBayes())
file.append(demo.LinearSVM())
# file2.append(demo.GaussianSVM())
file.append(demo.OR())
file.append(demo.AND())
file.append(demo.MRC())
file.append(demo.S1())
file.append(demo.S2())
file.append(demo.S3())
file.append(demo.S4())
# file.append(demo.S5())
# file.append(demo.S6())
# file.append(demo.S7())
# file.append(demo.S8())
# file.append(demo.RandomForest())
file.append(demo.KNN())
# file.append(demo.XGBoost())
# file.append(demo.CatBoost())
# file.append(demo.ADABoost())

# file.append(demo.DecisionTree())
file2.sort(key=lambda x:x[2],reverse=True)
file.sort(key=lambda x:x[2],reverse=True)

if(file!=[]):
    plt.rcParams['figure.figsize'] = [9,5]
    for [fpr,tpr,auc,type,colour,marker,markevery] in file:
        plt.plot(fpr, tpr, color=colour,marker=marker,markevery=0.1,
                 ms=5,linewidth=1,label='%0.4f %s' %(auc,type))
    plt.title('ROC Curve',fontsize=14)
    plt.grid() 
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # plt.xticks([x/100 for x in range(0,100,5)])
    plt.ylabel('Probability of Detection',fontsize=14)
    plt.xlabel('Pobability of False Alarm',fontsize=14)
    plt.legend(title="AUC Values",loc = 'best',prop={'size':13})
    plt.show()   
    # plt.clf()
    
if(file2!=[]):
    plt.rcParams['figure.figsize'] = [9, 5]
    for [fpr,tpr,auc,type,colour,marker,markevery] in file2:
        plt.plot(fpr, tpr, color=colour,marker=marker,markevery=0.1,ms=5,
                 label='%0.4f %s' %(auc,type))
    plt.title('ROC Curve')
    plt.grid()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # plt.xticks([x/100 for x in range(0,45,5)])
    plt.ylabel('Probability of Detection')
    plt.xlabel('Pobability of False Alarm')
    plt.legend(title="AUC Values",loc = 'best')
    plt.show()
    plt.clf()

    
# =============================================================================
#     ## AUC vs sensing time
# 
#     datasetsize=[3,4,5,6,7]
#     auclinearsvm=   [0.9083,0.9272,0.9376,0.9432,0.9511]
#     aucgaussiansvm= [0.9061,0.9285,0.9353,0.9438,0.9506]
#     aucmlp=         [0.9060,0.9251,0.9376,0.9424,0.9498]
#     auclogistic=    [0.9043,0.9248,0.9373,0.9428,0.9509]
#     aucmrc=         [0.8991,0.9165,0.9252,0.9315,0.9377]
#     aucor=          [0.8934,0.9149,0.9258,0.9351,0.9423]
#     aucnb =         [0.8922,0.9044,0.9250,0.9373,0.9431]
#     aucand=         [0.7065,0.7255,0.7357,0.7428,0.7582]
#     
#     plt.rcParams['figure.figsize'] = [12, 8]
#     
#     plt.plot(datasetsize, auclinearsvm, color='fuchsia',marker="X",markevery=1,ms=5,
#              linewidth=1,label='LinearSVM')
#     plt.plot(datasetsize, aucgaussiansvm, color='black',marker=2,markevery=1,ms=5,
#              linewidth=1,label='GaussianSVM')
#     plt.plot(datasetsize, aucmlp, color='darkgreen',marker="$c$",markevery=1,ms=5,
#              linewidth=1,label='MLP')
#     plt.plot(datasetsize, auclogistic, color='gold',marker="o",markevery=1,ms=5,
#              linewidth=1,label='Logistic')
#     plt.plot(datasetsize, aucmrc, color='lightcoral',marker="<",markevery=1,ms=5,
#              linewidth=1,label='MRC')
#     plt.plot(datasetsize, aucor, color='darkred',marker="v",markevery=1,ms=5,
#              linewidth=1,label='OR')
#     plt.plot(datasetsize, aucnb, color='purple',marker="*",markevery=1,ms=5,
#              linewidth=1,label='NaiveBayes')
#     plt.plot(datasetsize, aucand, color='brown',marker=">",markevery=1,ms=5,
#              linewidth=1,label='AND')
#     
#     plt.title('AUC vs Sensing Time')
#     plt.grid()
#     plt.legend(loc = 'lower right')
#     plt.xlim([2, 8])
#     plt.ylim([0.70, 1])
#     plt.ylabel('AUC Value')
#     plt.xlabel('Sensing Time (microseconds)')
#     plt.show()
#      
# =============================================================================
    
    ## AUC vs SU number

    datasetsize=[2,3,4,5]
    auclinearsvm=   [0.9125, 0.9364, 0.9537, 0.9537]
    aucKNN=         [0.8995, 0.9212, 0.9349, 0.9349]
    aucmlp=         [0.9039, 0.9277, 0.9531, 0.9549]
    auclogistic=    [0.9124, 0.9292, 0.9528, 0.9528]
    # aucmrc=         [0.9130, 0.9252, 0.9446, 0.9600, 0.9703, 0.9785]
    # aucor=          [0.9070, 0.9258, 0.9454, 0.9574, 0.9669, 0.9733]
    aucnb =         [0.8934, 0.9194, 0.9472, 0.9472]
    #auccatboost=   [0.9048, 0.9218, 0.9481, 0.9612, 0.9726, 0.9801]
    
    plt.rcParams['figure.figsize'] = [9, 5]
    plt.plot(datasetsize, auclinearsvm, color='fuchsia',marker="X",markevery=1,ms=5,
             linewidth=1,label='SVM')
    plt.plot(datasetsize, aucKNN, color='black',marker=2,markevery=1,ms=5,
              linewidth=1,label='KNN')
    plt.plot(datasetsize, aucmlp, color='darkgreen',marker="$c$",markevery=1,ms=5,
             linewidth=1,label='MLP')
    plt.plot(datasetsize, auclogistic, color='gold',marker="o",markevery=1,ms=5,
             linewidth=1,label='Logistic')
    # plt.plot(datasetsize, aucmrc, color='lightcoral',marker="<",markevery=1,ms=5,
    #          linewidth=1,label='MRC')
    # plt.plot(datasetsize, aucor, color='darkred',marker="v",markevery=1,ms=5,
    #          linewidth=1,label='OR')
    plt.plot(datasetsize, aucnb, color='purple',marker="*",markevery=1,ms=5,
              linewidth=1,label='NBC')
    # plt.plot(datasetsize, auccatboost, color='pink',marker="d",markevery=1,ms=5,
    #          linewidth=1,label='CatBoost')
    
    # plt.title('AUC vs SU number',fontsize=14)
    plt.grid()
    plt.legend(loc = 'best',prop={'size':13})
    plt.xlim([2, 5])
    # plt.ylim([0.88, 1])
    plt.ylabel('AUC Value',fontsize=14)
    plt.xlabel('Number of SU',fontsize=14)
    plt.show()
              
# C()