#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
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
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
)
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as score
import pydot
import os
from imblearn.over_sampling import SMOTE

from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions

# get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import display, HTML

pd.options.display.max_rows = None
pd.options.display.max_columns = None


# In[ ]:


testData = "dataset/test.csv"
dataA = pd.read_csv(testData, sep=",", index_col="ID_code")
dfTest = pd.DataFrame(dataA)

trainData = "dataset/train.csv"
dataB = pd.read_csv(trainData, sep=",", index_col="ID_code")
dfTrain = pd.DataFrame(dataB)

sample_submission = "dataset/sample_submission.csv"
dataC = pd.read_csv(sample_submission, sep=",", index_col="ID_code")
df_sample = pd.DataFrame(dataC)


# In[ ]:


y_train_complete = dfTrain.iloc[:, 0:1]
X_train_complete = dfTrain.drop(["target"], axis=1)

print(y_train_complete.shape)
print(X_train_complete.shape)


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(
    X_train_complete.values, y_train_complete.values, test_size=0.1, random_state=12
)


# In[ ]:


sm = SMOTE(random_state=12, ratio=1.0)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)


# In[1]:


# Classifier
bayes_cv_tuner = BayesSearchCV(
    estimator=lgb.LGBMRegressor(
        objective="binary",
        metric="auc",
        tree_method="approx",
        learning_rate=0.1,
        n_jobs=-1,
        silent=1,
        #         n_estimators= 7000,
        verbose=0,
    ),
    search_spaces={
        "max_delta_step": (0, 20),
        "max_depth": (3, 15),
        "max_delta_step": (0, 20),
        "min_child_samples": (0, 50),
        "max_bin": (100, 1000),
        "subsample": (0.01, 1.0, "uniform"),
        "subsample_freq": (0, 10),
        "colsample_bytree": (0.01, 1.0, "uniform"),
        "colsample_bylevel": (0.01, 1.0, "uniform"),
        "min_child_weight": (10, 20),
        "gamma": (1e-2, 0.5, "log-uniform"),
        "subsample_for_bin": (100000, 500000),
        "reg_lambda": (1e-4, 1000, "log-uniform"),
        "reg_alpha": (1e-4, 1.0, "log-uniform"),
        "scale_pos_weight": (1e-6, 500, "log-uniform"),
        "n_estimators": (5000, 10000),
    },
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    n_jobs=3,
    n_iter=50,
    verbose=0,
    refit=True,
    random_state=42,
)


def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)
    # Get current parameters and the best parameters
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print(
        "Model #{}\nBest ROC-AUC: {}\nBest params: {}\n".format(
            len(all_models),
            np.round(bayes_cv_tuner.best_score_, 4),
            bayes_cv_tuner.best_params_,
        )
    )
    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name + "_cv_results.csv")


# In[ ]:


# Fit the model
result = bayes_cv_tuner.fit(x_train_res, y_train_res, callback=status_print)


# In[ ]:


lgbm_pred = result.predict(x_val)


# In[ ]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test_dn.values, lgbm_pred.round())
print(cm)
# Accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test_dn.values, lgbm_pred.round())
print(accuracy)


# In[ ]:


pred = pd.DataFrame(
    lgbm_pred.round(), index=df_sample.index, columns=dfTrain.columns[0:1]
)
pred.to_csv("lgbm_UPSampling.csv", sep=",")


# In[ ]:


# In[ ]:
