# from matplotlib.cbook import to_filehandle
import numpy as np
# from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
# from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from scipy import special
import tensorflow as tf
tf.autograph.set_verbosity(1)



class Classification:
    # print(np.concatenate((y_pred.reshpipape(len(y_pred),1), 
    #                       self.y_test.reshape(len(self.y_test),1)),1))
    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None, Samples=50,
                 SU=3, X_test_2=None, SNR=None):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.Samples=Samples
        self.SU=SU
        self.X_test_2=X_test_2
        self.SNR=SNR
        # sc= StandardScaler()
        # self.X_train = sc.fit_transform(self.X_train) 
        # self.X_test = sc.transform(self.X_test)
        self.X_combined = np.r_[self.X_train, self.X_test]
        self.y_combined = np.r_[self.y_train, self.y_test] 
        # self.y_train=self.y_train.reshape(-1)
        # df_train.info()
        
    def main(self):
        val=[]
      
    def MLP(self):
        type='MLP'
        marker="s"
        ann = tf.keras.models.Sequential()

        ann.add(tf.keras.layers.Dense(units=self.SU, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=len(self.y_train), activation='relu'))
        ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        ann.fit(self.X_train, self.y_train, epochs =500,verbose=0)
        y_pred2=ann.predict(self.X_test)
        y_pred2=y_pred2.flatten()
        fpr, tpr, _ = metrics.roc_curve(self.y_test,  y_pred2)
        auc = metrics.auc(fpr,tpr)
        return[fpr,tpr,auc,type,"darkgreen",marker,int((len(fpr))*0.037)]
    
        # return output
    
        
    
    def Logistic(self):
        classifier=LogisticRegression()
        type='Logistic'
        marker="o"
        parameters =[{'C': [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4], 'max_iter':[1000],
                      'solver': ['newton-cg','lbfgs','sag'], 'penalty': ['l2']},
                     {'C': [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4], 'max_iter':[1000],
                      'solver': ['saga','liblinear'], 'penalty': ['l1','l2']}]
        grid_search = GridSearchCV(estimator = classifier, param_grid = parameters,
                                   scoring = 'accuracy', n_jobs = -1, cv=10,verbose=0)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_
        best_parameters=grid_search.best_params_
        y_pred=grid_search.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters]
        y_pred2=grid_search.predict_proba(self.X_test)
        y_pred2=y_pred2[:,1]
        fpr, tpr, _ = metrics.roc_curve(self.y_test,  y_pred2)
        auc = metrics.auc(fpr,tpr)
        return[fpr,tpr,auc,type,"black",marker,int((len(fpr))*0.037)]
    
        # ROC(self.X_test,self.y_test, y_pred2)
        # ROC2(self.X_test,self.y_test, y_pred2)
        
        # return output
            
    def DecisionTree(self):
        classifier= DecisionTreeClassifier()
        type='DecisionTree'
        marker="c"
        parameters =[{'max_depth':[2,3,4,5,6],'criterion': ['gini','entropy'],
                      'min_samples_leaf':[1,2,3,4,5,6,7,8,9],
                      'min_samples_leaf':[0.001,0.0025,0.005,0.075,0.01],
                      'splitter':['best','random']}]
        grid_search = GridSearchCV(estimator = classifier,param_grid=parameters,
                                   scoring = 'accuracy',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_
        best_parameters=grid_search.best_params_
        y_pred=grid_search.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters]
    
        y_pred2=grid_search.predict_proba(self.X_test)
        y_pred2=y_pred2[:,1]
        fpr, tpr, _ = metrics.roc_curve(self.y_test,  y_pred2)
        auc = metrics.auc(fpr,tpr)
        return[fpr,tpr,auc,type,"gold",marker,int((len(fpr))*0.037)]
    
        # ROC(self.X_test,self.y_test, y_pred2)
        # ROC2(self.X_test,self.y_test, y_pred2)
        
        # return output
    
    def RandomForest(self):
        classifier=RandomForestClassifier()
        type='RandomForest'
        marker="p"
        parameters =[{'n_estimators':[10,50,100,250,500] ,'criterion': ['gini','entropy'],
                      'max_features':['log2','sqrt']}]
        grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,
                                   scoring = 'accuracy',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        # best_accuracy = grid_search.best_score_
        # best_parameters=grid_search.best_params_
        # y_pred=grid_search.predict(self.X_test)
        # cm = confusion_matrix(self.y_test, y_pred)
        # accuracy=accuracy_score(self.y_test, y_pred)
        # cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        # output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters]    
        
        y_pred2=grid_search.predict_proba(self.X_test)
        y_pred2=y_pred2[:,1]
        fpr, tpr, _ = metrics.roc_curve(self.y_test,  y_pred2)
        auc = metrics.auc(fpr,tpr)
        if(len(fpr)>500):
            val=(len(fpr))*0.037
        elif (len(fpr)>10):
            val=(len(fpr))*0.05
        else:
            val=1
        return[fpr,tpr,auc,type,"tan",marker,val]
    
        # ROC(self.X_test,self.y_test, y_pred2)
        # ROC2(self.X_test,self.y_test, y_pred2)
        
        # return output
    
    def KNN(self):
        classifier = KNeighborsClassifier()
        type='KNN'
        marker="x"
        parameters =[{'n_neighbors': [5,7,9,11], 'weights':['uniform','distance'], 'n_jobs':[-1]}]
        grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,
                                   scoring = 'accuracy',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_
        best_parameters=grid_search.best_params_
        y_pred=grid_search.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        plt=1
        # plt=plot_learning_curve(estimator=classifier,title=type,X=self.X_combined, 
        #                         y=self.y_combined)
        output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters,plt]
        
        y_pred2=grid_search.predict_proba(self.X_test)
        y_pred2=y_pred2[:,1]
        fpr, tpr, _ = metrics.roc_curve(self.y_test,  y_pred2)
        auc = metrics.auc(fpr,tpr)
        if(len(fpr)>1000):
            val=(len(fpr))*0.037
        else:
            val=1
        return[fpr,tpr,auc,type,"blue",marker,val]

        # return output
    
    def NaiveBayes(self):
        classifier = GaussianNB()
        type='NBC'
        marker="*"
        parameters =[{'var_smoothing':[1e-9]}]
        grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,
                                   scoring = 'accuracy',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_
        best_parameters=grid_search.best_params_
        y_pred=grid_search.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        plt=1
        # plt=plot_learning_curve(estimator=classifier,title=type,
        #                         X=self.X_combined, y=self.y_combined)
        output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters,plt]
        
        y_pred2=grid_search.predict_proba(self.X_test)
        y_pred2=y_pred2[:,1]
        fpr, tpr, _ = metrics.roc_curve(self.y_test,  y_pred2)
        auc = metrics.auc(fpr,tpr)
        return[fpr,tpr,auc,type,"red",marker,int((len(fpr))*0.037)]
    
        # return output
        
    def LinearSVM(self):      
        classifier=SVC()
        type='SVM'
        marker="|"
        parameters =[{'C': [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4], 'kernel': ['linear'],
                      'probability':[True]}]
        grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,
                                   scoring = 'accuracy',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_
        best_parameters=grid_search.best_params_
        y_pred=grid_search.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        plt=1
        # plt=plot_learning_curve(estimator=classifier,title=type,
        #                         X=self.X_combined, y=self.y_combined)
        output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters,plt]
        
        y_pred2=grid_search.predict_proba(self.X_test)
        y_pred2=y_pred2[:,1]
        fpr, tpr, _ = metrics.roc_curve(self.y_test,  y_pred2)
        auc = metrics.auc(fpr,tpr)
        return[fpr,tpr,auc,type,"fuchsia",marker,int((len(fpr))*0.037)]
    
        # return output
        
    def GaussianSVM(self):      
        classifier=SVC()
        type='GaussianSVM'
        marker=2
        parameters =[{'C': [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4], 'kernel': ['rbf'], 
                      'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 'scale'],
                      'probability':[True]}]
        grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,
                                   scoring = 'accuracy',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_
        best_parameters=grid_search.best_params_
        y_pred=grid_search.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        plt=1
        # plt=plot_learning_curve(estimator=classifier,title=type,
        #                         X=self.X_combined, y=self.y_combined)
        output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters,plt]
        
        y_pred2=grid_search.predict_proba(self.X_test)
        y_pred2=y_pred2[:,1]
        fpr, tpr, _ = metrics.roc_curve(self.y_test,  y_pred2)
        auc = metrics.auc(fpr,tpr)
        return[fpr,tpr,auc,type,"black",marker,int((len(fpr))*0.037)]
    
        # return output
    
        
    def XGBoost(self):
        classifier=XGBClassifier()
        type='XGBoost'
        marker="D"
        parameters =[{'n_jobs':[-1],'use_label_encoder':[False],
                      'eval_metric':['error','logloss', 'auc'], 
                      'objective':['binary:logistic'],
        'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 'max_depth': [1, 3, 5, 7, 9]}]
        grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,
                                   scoring = 'accuracy',n_jobs = -1, cv=10, verbose=0)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_
        best_parameters=grid_search.best_params_
        y_pred=grid_search.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        plt=1
        # plt=plot_learning_curve(estimator=classifier,title=type,
        #                         X=self.X_combined, y=self.y_combined)
        output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters,plt]
        
        y_pred2=grid_search.predict_proba(self.X_test)
        y_pred2=y_pred2[:,1]
        fpr, tpr, _ = metrics.roc_curve(self.y_test,  y_pred2)
        auc = metrics.auc(fpr,tpr)
        return[fpr,tpr,auc,type,"pink",marker,int((len(fpr))*0.037)]
    
        # return output
    

    def CatBoost(self):
        classifier=CatBoostClassifier()
        type='CatBoost'
        marker="d"
        parameters =[{'custom_loss':['AUC', 'Accuracy'], 'verbose':[False],
                      'allow_writing_files':[False] }]
        grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,
                                   scoring = 'accuracy',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_
        best_parameters=grid_search.best_params_
        y_pred=grid_search.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        plt=1
        # plt=plot_learning_curve(estimator=classifier,title=type,
        #                         X=self.X_combined, y=self.y_combined)
        output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters,plt]
        
        y_pred2=grid_search.predict_proba(self.X_test)
        y_pred2=y_pred2[:,1]
        fpr, tpr, _ = metrics.roc_curve(self.y_test,  y_pred2)
        auc = metrics.auc(fpr,tpr)
        return[fpr,tpr,auc,type,"lightgrey",marker,int((len(fpr))*0.037)]
    
        # return output
    
        
    def ADABoost(self):
        classifier=AdaBoostClassifier()
        type='ADABoost'
        marker="|"
        parameters =[{'n_estimators':[75], 'algorithm':['SAMME', 'SAMME.R']}]
        grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,
                                   scoring = 'accuracy',n_jobs = -1, cv=10)
        grid_search.fit(self.X_train, self.y_train)
        best_accuracy = grid_search.best_score_
        best_parameters=grid_search.best_params_
        y_pred=grid_search.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy=accuracy_score(self.y_test, y_pred)
        cm2=confusion_matrix(self.y_train, grid_search.predict(self.X_train))
        plt=1
        # plt=plot_learning_curve(estimator=classifier,title=type,
        #                         X=self.X_combined, y=self.y_combined)
        output=[type,accuracy*100,cm,cm2,best_accuracy*100,best_parameters,plt]
        
        y_pred2=grid_search.predict_proba(self.X_test)
        y_pred2=y_pred2[:,1]
        fpr, tpr, _ = metrics.roc_curve(self.y_test,  y_pred2)
        auc = metrics.auc(fpr,tpr)
        return[fpr,tpr,auc,type,"olive",marker,int((len(fpr))*0.037)]
    
        # return output
        
    def OR(self):
        Pfa_target=[x/10000.0 for x in range(25,10000,25)]
        tpr=[]
        fpr=[]
        type="OR"
        marker="v"
        for i in range(len(Pfa_target)):
            alpha=1-Pfa_target[i]
            lambd = 2*special.gammainccinv(self.Samples/2,Pfa_target[i])/self.Samples
            # lambd = 2*special.gammainccinv(self.SU/2,Pfa_target[i])/self.SU
            y_pred=np.array(np.sum(self.X_test>=lambd,1)>0, dtype=int)    
            
            tn=np.sum(np.logical_not(self.y_test)&np.logical_not(y_pred))
            tp=np.sum(self.y_test&y_pred)
            fn=np.sum(self.y_test&np.logical_not(y_pred))
            fp=np.sum(np.logical_not(self.y_test)&y_pred)
            tpr.append(tp/(tp+fn))
            fpr.append(fp/(fp+tn))
        
        auc = metrics.auc(fpr,tpr)
        return[fpr,tpr,auc,type,"darkred",marker,int((len(fpr))*0.01)]
    
    def AND(self):
        Pfa_target=[x/10000.0 for x in range(25,10000,25)]
        tpr=[]
        fpr=[]
        type="AND"
        marker=">"
        for i in range(len(Pfa_target)):
            alpha=1-Pfa_target[i]
            lambd = 2*special.gammainccinv(self.Samples/2,Pfa_target[i])/self.Samples
            # lambd = 2*special.gammainccinv(self.SU/2,Pfa_target[i])/self.SU
            y_pred=np.array(np.sum(self.X_test>=lambd,1)==self.SU, dtype=int)
            
            tn=np.sum(np.logical_not(self.y_test)&np.logical_not(y_pred))
            tp=np.sum(self.y_test&y_pred)
            fn=np.sum(self.y_test&np.logical_not(y_pred))
            fp=np.sum(np.logical_not(self.y_test)&y_pred)
            tpr.append(tp/(tp+fn))
            fpr.append(fp/(fp+tn))
        
        auc = metrics.auc(fpr,tpr)
        return[fpr,tpr,auc,type,"brown",marker,int((len(fpr))*0.01)]
    
    def MRC(self):
        Pfa_target=[x/10000.0 for x in range(25,10000,25)]
        tpr=[]
        fpr=[]
        type="MRC"
        marker="<"
        for i in range(len(Pfa_target)):
            alpha=1-Pfa_target[i]
            lambd = 2*special.gammainccinv(self.Samples/2,Pfa_target[i])/self.Samples
            # lambd = 2*special.gammainccinv(self.SU/2,Pfa_target[i])/self.SU
            y_pred=np.array(np.sum(self.X_test_2,1)>lambd, dtype=int)
            
            tn=np.sum(np.logical_not(self.y_test)&np.logical_not(y_pred))
            tp=np.sum(self.y_test&y_pred)
            fn=np.sum(self.y_test&np.logical_not(y_pred))
            fp=np.sum(np.logical_not(self.y_test)&y_pred)
            tpr.append(tp/(tp+fn))
            fpr.append(fp/(fp+tn))
        
        auc = metrics.auc(fpr,tpr)
        return[fpr,tpr,auc,type,"lightcoral",marker,int((len(fpr))*0.01)]
        
    def S1(self):
        Pfa_target=[x/10000.0 for x in range(25,10000,25)]
        tpr=[]
        fpr=[]
        type="S1"
        marker="1"
        for i in range(len(Pfa_target)):
            alpha=1-Pfa_target[i]
            lambd = 2*special.gammainccinv(self.Samples/2,Pfa_target[i])/self.Samples
            # lambd = 2*special.gammainccinv(self.SU/2,Pfa_target[i])/self.SU
            y_pred=np.array(self.X_test[:,0]>=lambd, dtype=int)
            
            tn=np.sum(np.logical_not(self.y_test)&np.logical_not(y_pred))
            tp=np.sum(self.y_test&y_pred)
            fn=np.sum(self.y_test&np.logical_not(y_pred))
            fp=np.sum(np.logical_not(self.y_test)&y_pred)
            tpr.append(tp/(tp+fn))
            fpr.append(fp/(fp+tn))
        
        auc = metrics.auc(fpr,tpr)
        return[fpr,tpr,auc,type,"midnightblue",marker,int((len(fpr))*0.01)]
        
    def S2(self):
        Pfa_target=[x/10000.0 for x in range(25,10000,25)]
        tpr=[]
        fpr=[]
        type="S2"
        marker="2"
        for i in range(len(Pfa_target)):
            alpha=1-Pfa_target[i]
            lambd = 2*special.gammainccinv(self.Samples/2,Pfa_target[i])/self.Samples
            # lambd = 2*special.gammainccinv(self.SU/2,Pfa_target[i])/self.SU
            y_pred=np.array(self.X_test[:,1]>=lambd, dtype=int)
            
            tn=np.sum(np.logical_not(self.y_test)&np.logical_not(y_pred))
            tp=np.sum(self.y_test&y_pred)
            fn=np.sum(self.y_test&np.logical_not(y_pred))
            fp=np.sum(np.logical_not(self.y_test)&y_pred)
            tpr.append(tp/(tp+fn))
            fpr.append(fp/(fp+tn))
        
        auc = metrics.auc(fpr,tpr)
        return[fpr,tpr,auc,type,"royalblue",marker,int((len(fpr))*0.01)]
    
    def S3(self):
        Pfa_target=[x/10000.0 for x in range(25,10000,25)]
        tpr=[]
        fpr=[]
        type="S3"
        marker="3"
        for i in range(len(Pfa_target)):
            alpha=1-Pfa_target[i]
            lambd = 2*special.gammainccinv(self.Samples/2,Pfa_target[i])/self.Samples
            # lambd = 2*special.gammainccinv(self.SU/2,Pfa_target[i])/self.SU
            y_pred=np.array(self.X_test[:,2]>=lambd, dtype=int)
            
            tn=np.sum(np.logical_not(self.y_test)&np.logical_not(y_pred))
            tp=np.sum(self.y_test&y_pred)
            fn=np.sum(self.y_test&np.logical_not(y_pred))
            fp=np.sum(np.logical_not(self.y_test)&y_pred)
            tpr.append(tp/(tp+fn))
            fpr.append(fp/(fp+tn))
        
        auc = metrics.auc(fpr,tpr)
        return[fpr,tpr,auc,type,"green",marker,int((len(fpr))*0.01)]
    
    def S4(self):
        Pfa_target=[x/10000.0 for x in range(25,10000,25)]
        tpr=[]
        fpr=[]
        type="S4"
        marker="4"
        for i in range(len(Pfa_target)):
            alpha=1-Pfa_target[i]
            lambd = 2*special.gammainccinv(self.Samples/2,Pfa_target[i])/self.Samples
            # lambd = 2*special.gammainccinv(self.SU/2,Pfa_target[i])/self.SU
            y_pred=np.array(self.X_test[:,2]>=lambd, dtype=int)
            
            tn=np.sum(np.logical_not(self.y_test)&np.logical_not(y_pred))
            tp=np.sum(self.y_test&y_pred)
            fn=np.sum(self.y_test&np.logical_not(y_pred))
            fp=np.sum(np.logical_not(self.y_test)&y_pred)
            tpr.append(tp/(tp+fn))
            fpr.append(fp/(fp+tn))
        
        auc = metrics.auc(fpr,tpr)
        return[fpr,tpr,auc,type,"orange",marker,int((len(fpr))*0.01)]
    
    def S5(self):
        Pfa_target=[x/10000.0 for x in range(25,10000,25)]
        tpr=[]
        fpr=[]
        type="S5"
        marker="d"
        for i in range(len(Pfa_target)):
            alpha=1-Pfa_target[i]
            lambd = 2*special.gammainccinv(self.Samples/2,Pfa_target[i])/self.Samples
            # lambd = 2*special.gammainccinv(self.SU/2,Pfa_target[i])/self.SU
            y_pred=np.array(self.X_test[:,2]>=lambd, dtype=int)
            
            tn=np.sum(np.logical_not(self.y_test)&np.logical_not(y_pred))
            tp=np.sum(self.y_test&y_pred)
            fn=np.sum(self.y_test&np.logical_not(y_pred))
            fp=np.sum(np.logical_not(self.y_test)&y_pred)
            tpr.append(tp/(tp+fn))
            fpr.append(fp/(fp+tn))
        
        auc = metrics.auc(fpr,tpr)
        return[fpr,tpr,auc,type,"red",marker,int((len(fpr))*0.01)]
    
    def S6(self):
        Pfa_target=[x/10000.0 for x in range(25,10000,25)]
        tpr=[]
        fpr=[]
        type="S6"
        marker="D"
        for i in range(len(Pfa_target)):
            alpha=1-Pfa_target[i]
            lambd = 2*special.gammainccinv(self.Samples/2,Pfa_target[i])/self.Samples
            # lambd = 2*special.gammainccinv(self.SU/2,Pfa_target[i])/self.SU
            y_pred=np.array(self.X_test[:,2]>=lambd, dtype=int)
            
            tn=np.sum(np.logical_not(self.y_test)&np.logical_not(y_pred))
            tp=np.sum(self.y_test&y_pred)
            fn=np.sum(self.y_test&np.logical_not(y_pred))
            fp=np.sum(np.logical_not(self.y_test)&y_pred)
            tpr.append(tp/(tp+fn))
            fpr.append(fp/(fp+tn))
        
        auc = metrics.auc(fpr,tpr)
        return[fpr,tpr,auc,type,"red",marker,int((len(fpr))*0.01)]
        
        
      
# =============================================================================
# def plot_learning_curve(estimator,title,X,y,cv=10,n_jobs=-1,train_sizes=np.linspace(0.1, 1.0, 10)):
#     y=y.reshape(-1)
#     fig, axes = plt.subplots(3, 1, figsize=(10, 10))
#     axes[0].set_title(title)
#     axes[0].set_xlabel("Training examples")
#     axes[0].set_ylabel("Score")
# 
#     train_sizes, train_scores, test_scores, fit_times,\
#         _ = learning_curve(estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes,
#                            return_times=True)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     fit_times_mean = np.mean(fit_times, axis=1)
#     fit_times_std = np.std(fit_times, axis=1)
# 
#     axes[0].grid()
#     axes[0].fill_between(train_sizes,train_scores_mean - train_scores_std,
#                          train_scores_mean + train_scores_std,alpha=0.1,color="r")
#     axes[0].fill_between(train_sizes,test_scores_mean - test_scores_std,
#                          test_scores_mean + test_scores_std,alpha=0.1,color="g")
#     axes[0].plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
#     axes[0].plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
#     axes[0].legend(loc="best")
# 
#     axes[1].grid()
#     axes[1].plot(train_sizes, fit_times_mean, "o-")
#     axes[1].fill_between(train_sizes,fit_times_mean - fit_times_std,
#                          fit_times_mean + fit_times_std,alpha=0.1,)
#     axes[1].set_xlabel("Training examples")
#     axes[1].set_ylabel("fit_times")
#     axes[1].set_title("Scalability of the model")
# 
#     fit_time_argsort = fit_times_mean.argsort()
#     fit_time_sorted = fit_times_mean[fit_time_argsort]
#     test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
#     test_scores_std_sorted = test_scores_std[fit_time_argsort]
#     axes[2].grid()
#     axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
#     axes[2].fill_between(fit_time_sorted,test_scores_mean_sorted - test_scores_std_sorted,
#                          test_scores_mean_sorted + test_scores_std_sorted,alpha=0.1,)
#     axes[2].set_xlabel("fit_times")
#     axes[2].set_ylabel("Score")
#     axes[2].set_title("Performance of the model")
#     plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.4,hspace=0.4)
#     plt.show()
#     return plt
#     # cv = ShuffleSplit(n_splits=50, test_size=0.2)
# =============================================================================
    


# =============================================================================
# def ROC2(X_test,y_test, y_pred2):
#     y_pred2=y_pred2[:,1]
#     fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred2)
#     auc = metrics.auc(fpr,tpr)
#     
#     pd1=[]
#     pd2=[]
#     pd3=[]
#     
#     Pfa_target=[x/10000.0 for x in range(25,10000,25)]
#     tpr_s1=[]
#     fpr_s1=[]
#     tpr_s2=[]
#     fpr_s2=[]
#     tpr_s3=[]
#     fpr_s3=[]
#     for i in range(len(Pfa_target)):
#         alpha=1-Pfa_target[i]
#         lambd = 2*special.gammainccinv(25,Pfa_target[i])/50
#         # lambd = 2*special.gammainccinv(3/2,Pfa_target[i])/3
#         
#         pd1.append(special.gammainc(lambd*50/(2*1.276386), 25))
#         pd2.append(special.gammainc(lambd*50/(2*1.249404), 25))
#         pd3.append(special.gammainc(lambd*50/(2*1.079713), 25))
#         
#         #S1
#         y_pred_s1=np.array(X_test[:,0]>=lambd, dtype=int)
#         tn_s1=0
#         tp_s1=0
#         fn_s1=0
#         fp_s1=0
#         for i in range(len(y_test)):
#             if y_test[i]==0 and y_pred_s1[i]==0:
#                 tn_s1+=1
#             elif y_test[i]==1 and y_pred_s1[i]==1:
#                 tp_s1+=1
#             elif y_test[i]==1 and y_pred_s1[i]==0:
#                 fn_s1+=1
#             else:
#                 fp_s1+=1
#         tpr_s1.append(tp_s1/(tp_s1+fn_s1))
#         fpr_s1.append(fp_s1/(fp_s1+tn_s1))
#         
#         #S2
#         y_pred_s2=np.array(X_test[:,1]>=lambd, dtype=int)
#         tn_s2=0
#         tp_s2=0
#         fn_s2=0
#         fp_s2=0
#         for i in range(len(y_test)):
#             if y_test[i]==0 and y_pred_s2[i]==0:
#                 tn_s2+=1
#             elif y_test[i]==1 and y_pred_s2[i]==1:
#                 tp_s2+=1
#             elif y_test[i]==1 and y_pred_s2[i]==0:
#                 fn_s2+=1
#             else:
#                 fp_s2+=1
#         tpr_s2.append(tp_s2/(tp_s2+fn_s2))
#         fpr_s2.append(fp_s2/(fp_s2+tn_s2))
#         
#         #S3
#         y_pred_s3=np.array(X_test[:,2]>=lambd, dtype=int)
#         tn_s3=0
#         tp_s3=0
#         fn_s3=0
#         fp_s3=0
#         for i in range(len(y_test)):
#             if y_test[i]==0 and y_pred_s3[i]==0:
#                 tn_s3+=1
#             elif y_test[i]==1 and y_pred_s3[i]==1:
#                 tp_s3+=1
#             elif y_test[i]==1 and y_pred_s3[i]==0:
#                 fn_s3+=1
#             else:
#                 fp_s3+=1
#         tpr_s3.append(tp_s3/(tp_s3+fn_s3))
#         fpr_s3.append(fp_s3/(fp_s3+tn_s3))
#         
#     
#     
#     PFA=[]
#     PFA.append(Pfa_target)
#     PFA=np.transpose(PFA)
#     plt.title('ROC Curve')
#     plt.grid()
#     plt.plot(PFA, pd1, 'r',       label="Theoretical PD1: %0.4f"%metrics.auc(PFA,pd1))
#     plt.plot(fpr_s1, tpr_s1, 'r--', label="Practical PD1: %0.4f"%metrics.auc(fpr_s1,tpr_s1))
#     plt.plot(PFA, pd2, 'g',       label="Theoretical PD2: %0.4f"%metrics.auc(PFA,pd2))
#     plt.plot(fpr_s2, tpr_s2, 'g--', label="Practical PD2: %0.4f"%metrics.auc(fpr_s2,tpr_s2))
#     plt.plot(PFA, pd3, 'b',       label="Theoretical PD3: %0.4f"%metrics.auc(PFA,pd3))
#     plt.plot(fpr_s3, tpr_s3, 'b--', label="Practical PD3: %0.4f"%metrics.auc(fpr_s3,tpr_s3))
#     plt.plot(fpr, tpr, 'y', label = 'NB= %0.4f' %auc)
#     
#     plt.legend(loc = 'lower right')
#     # plt.xlim([0, 0.4])
#     # plt.ylim([0.6, 1])
#     # plt.xticks([x/100 for x in range(0,45,5)])
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.ylabel('Probability of Detection')
#     plt.xlabel('Pobability of False Alarm')
#     plt.show()
# =============================================================================
