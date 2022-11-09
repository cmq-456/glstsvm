# -*- coding: utf-8 -*-

import sys
sys.path.append('../../')
from libtsvm.estimators import LSTSVM
from libtsvm.model_selection import Validator
import pandas as pd
import numpy as np
import time
# UCI
X_train_N = pd.read_csv(r'F:\uci\X_train_N.csv')
X_train = X_train_N.iloc[0:540, 0:14].values

y_train = pd.read_csv(r'F:\uci\y_train.csv')
y_train = y_train.iloc[0:1, 0:540].values
y_train = y_train.ravel()

X_test_N = pd.read_csv(r'F:\uci\X_test_N.csv')
X_test = X_test_N.iloc[0:150, 0:14].values

y_test = pd.read_csv(r'F:\uci\y_test.csv')
y_test = y_test.iloc[0:1, 0:150].values
y_test = y_test.ravel()

W11 = pd.read_csv(r'F:\uci\W11.csv')
W11 = W11.iloc[0:240, 0:240].values


W22 = pd.read_csv(r'F:\uci\W22.csv')
W22 = W22.iloc[0:300, 0:300].values


# Step 2: Choose a TSVM-based estimator
kernel = 'linear'
lstsvm_clf = LSTSVM(kernel=kernel)

# Step 3: Evaluate the estimator using train/test split
eval_method = 't_t_split'

val = Validator(X_train, X_test, y_train, y_test, W,  W11, W22, eval_method, lstsvm_clf)
eval_func = val.choose_validator()

# Hyper-parameters of the classifier
h_params = {'C1': 2**0, 'C2': 2**1}

start = time.time()
acc, std, full_report = eval_func(h_params)
end = time.time()

print(f"Running time:{(end - start)}")

print("Accuracy: %.2f" % acc)
print(full_report)
