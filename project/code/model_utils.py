# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    validation_curve,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import precision_recall_curve

import scikitplot as skplt

from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold


def cross_validate_model(
    X, y, clf, search_spaces, scoring="roc_auc", n_folds=3, n_iter=20
):
    def status_print(optim_result):
        """Status callback durring bayesian hyperparameter search"""
        # Get all the models tested so far in DataFrame format
        all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)
        # Get current parameters and the best parameters
        best_params = pd.Series(bayes_cv_tuner.best_params_)
        print(
            f"Model #{len(all_models)}\n Best {scoring}: {np.round(bayes_cv_tuner.best_score_, 4)}\nBest params: {bayes_cv_tuner.best_params_}\n"
        )
        # Save all model results
        clf_name = bayes_cv_tuner.estimator.__class__.__name__
        all_models.to_csv(clf_name + "_cv_results.csv")

    # create bayesian optimizer
    bayes_cv_tuner = BayesSearchCV(
        estimator=clf,
        search_spaces=search_spaces,
        scoring=scoring,
        cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42),
        n_jobs=4,
        n_iter=n_iter,
        verbose=0,
        refit=True,
        random_state=786,
    )
    result = bayes_cv_tuner.fit(X, y, callback=status_print)

    return result



def benchmark_model_performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)
    
    
    # calculate AUC
    roc_auc = roc_auc_score(y_test, y_pred_probs[:,1])
    print(f'AUC: {roc_auc:.3f}')
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs[:,1])
    
    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs[:,1])
    # calculate precision-recall AUC
    pr_auc = auc(recall, precision)
    print(f'AUC: {pr_auc:.3f}')
    
    # print classification report
    print(classification_report(y_test, y_pred))
    
    
#    # setup figure
    fig = plt.figure(figsize=(8,12))
#    
#    # plot auc_roc figure
    ax = fig.add_subplot(211)
#    ax.plot([0, 1], [0, 1], linestyle='--')
#    ax.plot(fpr, tpr, marker='.')
#    ax.annotate('some text',xy=(0.9,0.1))
    
    skplt.metrics.plot_roc(y_test, y_pred_probs)
    
    # plot precision recall figure
    ax = fig.add_subplot(212)
    ax.plot([0, 1], [0.5, 0.5], linestyle='--')
#    ax.plot(recall, precision, marker='.')
    plt.show()
    
    return plt.gcf()



if __name__ == "__main__":
    # import dataset
    test_data = "dataset/test.csv"
    df_test = pd.read_csv(test_data, sep=",", index_col="ID_code")

    train_data = "dataset/train.csv"
    df_train = pd.read_csv(train_data, sep=",", index_col="ID_code")

    from sklearn.utils import resample

    # Separate majority and minority classes
    df2_majority = df_train[df_train["target"] == 0]
    df2_minority = df_train[df_train["target"] == 1]
    n_samples = df2_minority.target.sum()

    df2_majority_downsampled = resample(
        df2_majority, replace=False, n_samples=n_samples, random_state=99
    )
    df_downsampled = pd.concat([df2_majority_downsampled, df2_minority])
    X_dn = df_downsampled.drop(["target"], axis=1)
    y_dn = df_downsampled["target"]

    X_train_dn, X_test_dn, y_train_dn, y_test_dn = train_test_split(
        X_dn, y_dn, test_size=0.83, random_state=101
    )

    '''
    EXAMPLE USAGE OF MODULE
    =============================================
    '''

    from sklearn.neighbors import KNeighborsClassifier

    neigh = KNeighborsClassifier()
    search_params = {"n_neighbors": (1, 2300), "p": (1, 5), "weights": ("uniform", "distance")}

    cv_obj = cross_validate_model(
        X_train_dn,
        y_train_dn,
        neigh,
        search_spaces=search_params,
        scoring="roc_auc",
        n_iter=30,
    )

    model = cv_obj.best_estimator_
    model.fit(X_train_dn, y_train_dn)
    print(model)
    
    y_pred = model.predict(X_test_dn)
    print(classification_report(y_test_dn, y_pred))
