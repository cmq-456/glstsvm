# -*- coding: utf-8 -*-

import sys
sys.path.append('../../')
from libtsvm.estimators import LSTSVM
from libtsvm.mc_scheme import OneVsOneClassifier
from libtsvm.model_selection import Validator
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import matplotlib.pylab as pl

X_train_N = pd.read_csv(r'F:\data\X_train_N.csv')  
X_train = X_train_N.iloc[0:600, 0:11].values

y_train = pd.read_csv(r'F:\data\y_train.csv')
y_train = y_train.iloc[0:1, 0:600].values
y_train = y_train.ravel()

X_test_N = pd.read_csv(r'F:\data\X_test_N.csv')
X_test = X_test_N.iloc[0:200, 0:11].values

y_test = pd.read_csv(r'F:\data\y_test.csv')
y_test = y_test.iloc[0:1, 0:200].values
y_test = y_test.ravel()

W1_D = pd.read_csv(r'F:\data\W1_D.csv')
W1 = W1_D.iloc[0:150, 0:150].values

W2_D = pd.read_csv(r'F:\data\W2_D.csv')
W2 = W2_D.iloc[0:150, 0:150].values

W3_D = pd.read_csv(r'F:\data\W3_D.csv')
W3 = W3_D.iloc[0:150, 0:150].values

W4_D = pd.read_csv(r'F:\data\W4_D.csv')
W4 = W4_D.iloc[0:150, 0:150].values

W = np.row_stack((W1, W2, W3, W4))

# Step 2: Choose a TSVM-based estimator
kernel = 'RBF'
Glstsvm_clf = GLSTSVM(kernel=kernel)

# Step 3: Select a multi-class approach
ovo_Glstsvm = OneVsOneClassifier(Glstsvm_clf)


eval_method = 't_t_split' # Train/Test split


val = Validator(X_train, X_test, y_train, y_test, W, W11, W22, eval_method, ovo_Glstsvm)
eval_func = val.choose_validator()


# Hyper-parameters of the classifier

h_params = {'C1': 2**-1, 'C2': 2**-8, 'gamma': 2**-1}

start = time.time()
acc, std, full_report = eval_func(h_params)
end = time.time()

print(f"Running time:{(end - start)}")

print("Accuracy: %.2f" % acc)
print("Std: %.2f" % std)
print(full_report)


scatter=plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30, facecolors='none', zorder=10)

handles,labels=scatter.legend_elements(prop="colors")
labels=["Sample cluster 1","Sample cluster 2","Sample cluster 3","Sample cluster 4"]
legend=plt.legend(handles,labels,framealpha=0)
plt.xlabel('Kurtosis')
plt.ylabel('Standard deviation')
cm_light = mpl.colors.ListedColormap([ '#A0FFA0' , '#FFA0A0' , '#A0A0FF', '#A000EE'])


x_min, x_max = X_train[:, 0].min() - .1, X_train[:, 0].max() + .1
y_min, y_max = X_train[:, 1].min() - .1, X_train[:, 1].max() + .1

XX, YY = np.mgrid[x_min:x_max:3000j, y_min:y_max:3000j]

Z = val.estimator_predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)

plt.pcolormesh(XX, YY, Z, shading='auto', cmap = cm_light)
plt.contourf(XX, YY, Z, cmap=pl.cm.coolwarm, alpha=0.8)

plt.show()
