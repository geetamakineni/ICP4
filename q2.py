# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:29:28 2020

@author: geeta
"""

import numpy as np
import pandas as pd

glass=pd.read_csv('glass.csv')

X=glass.iloc[:,:-1].values
y=glass.iloc[:,9].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import classification_report
classification_report(y_test,y_pred)
