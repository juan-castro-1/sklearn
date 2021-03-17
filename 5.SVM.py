# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 09:05:24 2021

@author: juan_
"""

'SVM'
 
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris['data'][:, (2,3)] 
y = (iris['target'] == 2).astype(np.float64)

svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_svc', LinearSVC(C=1, loss='hinge')),
    ])

svm_clf.fit(X, y)

print(svm_clf.predict([[5.5,1.7]]))

## nonlinear SVM

from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_svm_clf = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),
    ('svm_clf', LinearSVC(C=10, loss='hinge'))
    ])
polynomial_svm_clf.fit(X, y)

## poly kernel

from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))
])

poly_kernel_svm_clf.fit(X, y)


## gaussian RBF kernel

rbf_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('scm_clf', SVC(kernel='rbf', gamma=5, C=0.001))
    ])
rbf_kernel_svm_clf.fit(X, y)

## svm reg

from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)

from sklearn.svm import SVR

svm_poly_reg = SVR(kernel='poly', degree= 2, C=100, epsilon = 0.1)
svm_poly_reg.fit(X, y)


## hyperparam GS

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline

irisdata = pd.read_csv('iris.csv')
irisdata = datasets.load_iris()

X = irisdata['data'][:, (2,3)] 
y = (irisdata['target'] == 2).astype(np.float64)


X.head()
X.info()

import seaborn as sns
sns.pairplot(irisdata,hue='class',palette='Dark2')

from sklearn.model_selection import train_test_split
X = irisdata.drop('class', axis=1)  
y = irisdata['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


kernels = ['Polynomial', 'RBF', 'Sigmoid','Linear']#A function which returns the corresponding SVC model
def getClassifier(ktype):
    if ktype == 0:
        # Polynomial kernal
        return SVC(kernel='poly', degree=8, gamma="auto")
    elif ktype == 1:
        # Radial Basis Function kernal
        return SVC(kernel='rbf', gamma="auto")
    elif ktype == 2:
        # Sigmoid kernal
        return SVC(kernel='sigmoid', gamma="auto")
    elif ktype == 3:
        # Linear kernal
        return SVC(kernel='linear', gamma="auto")

for i in range(4):
    # Separate data into test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Train a SVC model using different kernal
    svclassifier = getClassifier(i)
    svclassifier.fit(X_train, y_train) # Make pred
    y_pred = svclassifier.predict(X_test) # Evaluate
    print('Evaluation:', kernels[i], 'kernel')
    print(classification_report(y_test, y_pred))

    
## calibrando

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1,10,100], 'gamma': [1,0.1,0.01,0.001],
              'kernel': ['rbf', 'poly', 'sigmoid']    
    }

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)

print(grid.best_estimator_)

grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))#Output








