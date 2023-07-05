from Algorithms import Classification
# import numpy as np
import matplotlib.pyplot as plt
# import copy
import simulate as sm

# Tt = 100e-6
# auc1=[]
# auc2=[]
# auc3=[]
# auc4=[]
# auc5=[]
file=[]
file2=[]

# =============================================================================
# samples_range=[]
# for i in range (10,101,5):
#     samples_range.append(i)
#     X_train,y_train,SNR=sm.MCS(realiz=250,T=i/10e6)
#     X_test,y_test,SNR=sm.MCS(realiz=50000,T=i/10e6)
#     Samples=i
#     SU=len(SNR)
#     demo=Classification(X_train=X_train,y_train=y_train,X_test=X_test,
#       y_test=y_test,Samples=Samples,SU=SU,SNR=SNR)
#     # _,_,auc,_,_,_,_=demo.OR()
#     # auc1.append(auc)
#     # _,_,auc,_,_,_,_=demo.GaussianSVM()
#     # auc2.append(auc)
#     # _,_,auc,_,_,_,_=demo.LinearSVM()
#     # auc3.append(auc)
#     # _,_,auc,_,_,_,_=demo.Logistic()
#     # auc4.append(auc)
#     # _,_,auc,_,_,_,_=demo.MRC()
#     # auc5.append(auc)
#     print(f"{i}th Samples Done")
# 
# =============================================================================

# =============================================================================
# CALCULATING THROUGHPUT
# tau = np.array(samples_range)/(w*2)
# Th1 = (Tt-tau)/Tt*np.array(auc1)
# Th2 = (Tt-tau)/Tt*np.array(auc2)
# Th3 = (Tt-tau)/Tt*np.array(auc3)
# # Th4 = (Tt-tau)/Tt*np.array(auc4)
# # Th5 = (Tt-tau)/Tt*np.array(auc5)
# 
# 
# plt.rcParams['figure.figsize'] = [9, 5] 
# plt.title('Throughput vs AUC value')
# 
# plt.plot(auc2,Th2,label="GaussianSVM",marker="*")
# plt.plot(auc3,Th3,label="LinearSVM",marker="o")
# plt.plot(auc1,Th1,label="OR",marker="+")
# # plt.plot(auc4,Th4,label="Logistic",marker="1")
# # plt.plot(auc5,Th5,label="MRC",marker="2")
# # plt.plot(samples_range,auc2,label="GaussianSVM",marker="*")
# # plt.plot(samples_range,auc1,label="OR",marker="+")
# plt.ylabel('Throughput')
# plt.xlabel('AUC Value')
# plt.legend(loc = 'best')
# plt.grid()
# plt.show()
# 
# # plt.clf()
# =============================================================================


# CALCULATING ROC Pd vs Pf
X_train,y_train,SNR=sm.MCS(realiz=250)
X_test,y_test,SNR=sm.MCS(realiz=50000)
Samples = 50
SU = 3

demo=Classification(X_train=X_train,y_train=y_train,X_test=X_test,
                    y_test=y_test,Samples=Samples,SU=SU,SNR=SNR)

# file2.append(demo.Logistic())
# file2.append(demo.MLP())
# file2.append(demo.NaiveBayes())
file2.append(demo.LinearSVM())
# file2.append(demo.GaussianSVM())
# file2.append(demo.OR())
# file2.append(demo.AND())
# file2.append(demo.MRC())
file.append(demo.S1())
file.append(demo.S2())
file.append(demo.S3())
# file.append(demo.RandomForest())
# file.append(demo.KNN())
# file.append(demo.XGBoost())
# file.append(demo.CatBoost())
# file.append(demo.ADABoost())
# file.append(demo.DecisionTree())

# PLOTTING ROC CURVE Pd vs PF
if(file!=[]):
    plt.rcParams['figure.figsize'] = [9,5]
    for [fpr,tpr,auc,type,colour,marker,markevery] in file:
        plt.plot(fpr, tpr, color=colour,marker=marker,markevery=0.1,
                 ms=5,linewidth=1,label='%0.4f %s' %(auc,type))
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
