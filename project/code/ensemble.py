#!/usr/bin/env python
# coding: utf-8

# In[64]:


## Importing core libraries
import numpy as np
import pandas as pd
from time import time
import pprint
from imblearn.over_sampling import SMOTE

# Suppressing warnings because of skopt verbosity
import warnings

warnings.filterwarnings("ignore")
from sklearn.utils import resample
from sklearn.preprocessing import scale

# Our example dataset
from sklearn.datasets import load_boston

#
## Classifiers
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import xgboost as xgb

#
## Hyperparameters distributions
from scipy.stats import randint
from scipy.stats import uniform

#
## Model selection
from sklearn.model_selection import StratifiedKFold

#
## Metrics
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error
from sklearn.metrics import make_scorer

# Skopt functions
from skopt import BayesSearchCV
from skopt import gp_minimize  # Bayesian optimization using Gaussian Processes
from skopt.space import Real, Categorical, Integer
from skopt.utils import (
    use_named_args,
)  # decorator to convert a list of parameters to named arguments
from skopt.callbacks import (
    DeadlineStopper,
)  # Stop the optimization before running out of a fixed budget of time.
from skopt.callbacks import VerboseCallback  # Callback to control the verbosity
from skopt.callbacks import (
    DeltaXStopper,
)  # Stop the optimization If the last two positions at which the objective has been evaluated are less than delta

from sklearn.model_selection import StratifiedKFold

import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import neighbors, linear_model
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    validation_curve,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    make_scorer,
    roc_auc_score,
    accuracy_score,
    roc_curve,
    auc,
)
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as score
import pydot
import os
from statistics import mode

# Classifier
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
from time import time
from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions

testData = "dataset/test.csv"

dataA = pd.read_csv(testData, sep=",", index_col="ID_code")
dfTest = pd.DataFrame(dataA)


trainData = "dataset/train.csv"
dataB = pd.read_csv(trainData, sep=",", index_col="ID_code")
dfTrain = pd.DataFrame(dataB)


sample_submission = "dataset/sample_submission.csv"
dataC = pd.read_csv(sample_submission, sep=",", index_col="ID_code")
df_sample = pd.DataFrame(dataC)


y_train_complete = dfTrain.iloc[:, 0:1]
X_train_complete = dfTrain.drop(["target"], axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    X_train_complete, y_train_complete, test_size=0.20, random_state=42
)


model1 = lgb.LGBMRegressor(
    num_leaves=10,
    max_bin=119,
    min_data_in_leaf=11,
    learning_rate=0.02,
    bagging_fraction=1,
    min_sum_hessian_in_leaf=0.00245,
    bagging_freq=5,
    feature_fraction=0.05,
    lambda_l1=4.972,
    lambda_l2=2.276,
    min_gain_to_split=0.65,
    max_depth=14,
    save_binary=True,
    seed=1337,
    feature_fraction_seed=1337,
    bagging_seed=1337,
    drop_seed=1337,
    data_random_seed=1337,
    objective="binary",
    boosting_type="gbdt",
    verbose=1,
    boost_from_average=False,
    is_unbalance=True,
    metric="auc",
    random_state=42,
    n_jobs=-1,
    silent=False,
)


model2 = xgb.XGBClassifier(
    n_jobs=-1,
    colsample_bylevel=0.18273244758535206,
    colsample_bytree=0.7530295101748936,
    gamma=0.2941753333970994,
    max_delta_step=19,
    max_depth=7,
    min_child_weight=2,
    reg_alpha=0.1186916389388348,
    reg_lambda=0.03835472993899774,
    scale_pos_weight=0.5,
    subsample=0.6601228271348841,
    random_state=42,
    n_estimators=1000,
)

model3 = LogisticRegression()

print("LGBM\n===================================================================\n")
model1.fit(X_train, y_train)
print("XGBoost\n===================================================================\n")
model2.fit(X_train, y_train)
print("Logistic\n===================================================================\n")
model3.fit(X_train, y_train)

pred1 = model1.predict(X_test).round()
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)

final_pred = np.array([])


for i in range(0, len(X_test)):
    final_pred = np.append(final_pred, mode([pred1[i], pred2[i], pred3[i]]))


print("Results from the default parameters:")
print("Accuracy is ", accuracy_score(y_test_dn, final_pred) * 100)
print(confusion_matrix(y_test_dn, final_pred))
print(classification_report(y_test_dn, final_pred))


pred = pd.DataFrame(final_pred, index=df_sample.index, columns=dfTrain.columns[0:1])
pred.to_csv("Ensemble_xgb_lgbm_LogisticRegression(downsampledData).csv", sep=",")
