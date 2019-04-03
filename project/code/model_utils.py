# -*- coding: utf-8 -*-
import os
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
    X, y, clf, search_spaces, scoring="roc_auc", n_folds=3, n_iter=20, n_cpus=3
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
        n_jobs=n_cpus,
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
    roc_auc = roc_auc_score(y_test, y_pred_probs[:, 1])
    #    print(f'AUC: {roc_auc:.3f}')
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs[:, 1])

    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs[:, 1])
    # calculate precision-recall AUC
    pr_auc = auc(recall, precision)
    #    print(f'AUC: {pr_auc:.3f}')

    # print classification report
    report = "Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred))
    report = (
        report
        + "\n\nClassification Report:\n"
        + str(classification_report(y_test, y_pred))
    )

    #    # setup figure
    fig = plt.figure(figsize=(6, 10))
    #
    #    # plot auc_roc figure
    ax1 = fig.add_subplot(311)
    skplt.metrics.plot_roc(y_test, y_pred_probs, ax=ax1)

    # plot precision recall figure
    ax2 = fig.add_subplot(312)
    skplt.metrics.plot_precision_recall(y_test, y_pred_probs, ax=ax2)

    # plot stats
    ax3 = fig.add_subplot(313)
    ax3.text(0.0, 0.3, report, {"fontsize": 10}, fontproperties="monospace")
    ax3.axis("off")
    plt.tight_layout()

    #    plt.show()

    return (plt.gcf(), report)


if __name__ == "__main__":
    #    # import dataset
    #    test_data = "dataset/test.csv"
    #    df_test = pd.read_csv(test_data, sep=",", index_col="ID_code")
    #
    #    train_data = "dataset/train.csv"
    #    df_train = pd.read_csv(train_data, sep=",", index_col="ID_code")
    #
    #    from sklearn.utils import resample
    #
    #    # Separate majority and minority classes
    #    df2_majority = df_train[df_train["target"] == 0]
    #    df2_minority = df_train[df_train["target"] == 1]
    #    n_samples = df2_minority.target.sum()
    #
    #    df2_majority_downsampled = resample(
    #        df2_majority, replace=False, n_samples=n_samples, random_state=99
    #    )
    #    df_downsampled = pd.concat([df2_majority_downsampled, df2_minority])
    #    X_dn = df_downsampled.drop(["target"], axis=1)
    #    y_dn = df_downsampled["target"]
    #
    #    X_train_dn, X_test_dn, y_train_dn, y_test_dn = train_test_split(
    #        X_dn, y_dn, test_size=0.91, random_state=101
    #    )

    """
    EXAMPLE USAGE OF MODULE USING KNN
    =============================================
    """

    from sklearn.neighbors import KNeighborsClassifier

    neigh = KNeighborsClassifier()
    search_params = {
        "n_neighbors": (3, 1000),
        "p": (1, 4),
        "weights": ("uniform", "distance"),
    }

    cv_obj = cross_validate_model(
        X_train_dn,
        y_train_dn,
        neigh,
        search_spaces=search_params,
        scoring="roc_auc",
        n_iter=15,
    )

    # get the best estimator from cross validation
    model = cv_obj.best_estimator_
    model.fit(X_train_dn, y_train_dn)
    print(model)

    #    from sklearn.externals import joblib
    #    # Output a pickle file for the model
    #    joblib.dump(model, 'saved_model.pkl')
    #
    #    # Load the pickle file
    #    clf_load = joblib.load('saved_model.pkl')

    # bench mark the model
    stats_fig, report = benchmark_model_performance(model, X_test_dn, y_test_dn)
    stats_fig.savefig(
        os.path.join("preliminary-knn-using-downsampled.png"), dpi=300, format="png"
    )
    stats_fig.show()
