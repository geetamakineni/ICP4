# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:52:15 2020

@author: geeta
"""

import pandas as pd

train_df = pd.read_csv('train_preprocessed.csv')
test_df = pd.read_csv('test_preprocessed.csv')
correlation = train_df.corr(method='pearson')
correlation