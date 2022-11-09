
from libtsvm.estimators import LSTSVM
from libtsvm.model_selection import Validator, grid_search, save_result
from libtsvm.mc_scheme import OneVsOneClassifier

import pandas as pd
import numpy as np
# Step 1: Load dataset

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

W1_D = pd.read_csv(r'F:\data\W1.csv')
W1 = W1_D.iloc[0:150, 0:150].values

W2_D = pd.read_csv(r'F:\data\W2.csv')
W2 = W2_D.iloc[0:150, 0:150].values

W3_D = pd.read_csv(r'F:\data\W3.csv')
W3 = W3_D.iloc[0:150, 0:150].values

W4_D = pd.read_csv(r'F:\data\W4.csv')
W4 = W4_D.iloc[0:150, 0:150].values

W = np.row_stack((W1, W2, W3, W4))

# Step 2: Choose a TSVM-based estimator

kernel = 'RBF'
lstsvm_clf = GLSTSVM(kernel=kernel)

ovo_lstsvm = OneVsOneClassifier(lstsvm_clf)

# Step 3: Choose an evaluation method.
val = Validator(X_train, X_test, y_train, y_test, W, W11, W22, 't_t_split', ovo_lstsvm)
eval_method = val.choose_validator()


params = {'C1': (-8, 8), 'C2': (-8, 8), 'gamma': (-8, 8)}
best_acc, best_acc_std, opt_params, clf_results = grid_search(eval_method, params)

print("Best accuracy: %.2f+-%.2f | Optimal parameters: %s" % (best_acc, best_acc_std,
                                                                                  str(opt_params)))

# Step 5: Save the classification results
clf_type = 'multiclass' # Type of classification problem
save_result(val, clf_type, clf_results, 'GLSTSVM-RBF')