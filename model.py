import numpy as np
from sklearn import metrics as mt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def ml_model(X_train, y_train, X_test, y_test):
    classifier = SVC()
    type = 'LinearSVM'
    # marker = "X"
    parameters = [{'C': [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4],
                'kernel': ['linear'], 'probability':[True]}]
    grid_search = GridSearchCV(
        estimator=classifier, param_grid=parameters, scoring='accuracy', n_jobs=-1, cv=10)
    grid_search.fit(X_train, y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_
    y_pred = grid_search.predict(X_test)
    cm = mt.confusion_matrix(y_test, y_pred)
    accuracy = mt.accuracy_score(y_test, y_pred)
    fpr, tpr, _ = mt.roc_curve(y_test,  y_pred)
    # Create SVM classifier with linear kernel
    # clf = svm.SVC(kernel='linear')
    # # clf = svm.SVC(kernel='rbf') # gaussian

    # # Train the model using the training data
    # clf.fit(X_train, y_train)

    # # Make predictions on the test data
    # y_pred = clf.predict(X_test)

    # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

    auc = mt.auc(fpr, tpr)
    return fpr, tpr, auc

