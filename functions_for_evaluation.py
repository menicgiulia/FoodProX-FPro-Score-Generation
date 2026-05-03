import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.preprocessing import LabelBinarizer
from xgboost import XGBClassifier


def multiclass_roc_auc_score(y_test, y_probs, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    return roc_auc_score(y_test, y_probs, average=average)

def multiclass_average_precision_score(y_test, y_probs, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    return average_precision_score(y_test, y_probs, average=average)

def multiclass_roc_curve(y_test, y_probs):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    fpr = dict()
    tpr = dict()
    for i in range(y_probs.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_probs[:, i])
    return (fpr, tpr)

def multiclass_average_precision_curve(y_test, y_probs):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    precision = dict()
    recall = dict()
    for i in range(y_probs.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_probs[:, i])
    return (precision, recall)


def AUCAUPkfold_from_file(X, y, model_type, params_file, splits,
                          models_prefix, metrics_prefix, verbose=True):
    
    best_params = joblib.load(params_file)
    classes = np.unique(y)
    n_classes = len(classes)
    n_folds = len(splits)
    perf_auc = np.zeros((n_folds, n_classes))
    perf_aup = np.zeros((n_folds, n_classes))
    rocs = []
    prcs = []
    models = []

    for i, fold in enumerate(splits):
        tr, te = fold['train'], fold['test']

        if model_type == 'RF':
            clf = RandomForestClassifier(random_state=42, n_jobs=-1, 
                                         **best_params)

        elif model_type == 'XGB':
            clf = XGBClassifier(random_state=42,
                                use_label_encoder=False,
                                eval_metric='logloss',
                                tree_method='hist',
                                verbosity=0,
                                **best_params)

        else:
            raise ValueError(f"Unknown model_type {model_type!r}")

        clf.fit(X[tr], y[tr])
        y_proba = clf.predict_proba(X[te])

        if not np.array_equal(clf.classes_, classes):
            raise ValueError(
                f"Fold {i + 1}: classifier classes {clf.classes_} "
                f"do not match expected classes {classes}.")

        test_classes = np.unique(y[te])
        if not np.array_equal(test_classes, classes):
            raise ValueError(
                f"Fold {i + 1}: test set has classes {test_classes}, "
                f"expected {classes}. Per-class AUC/AUP cannot be computed safely.")

        models.append(clf)

        y_true = y[te]
        fold_rocs = []
        fold_prcs = []

        for ci, c in enumerate(classes):

            y_bin = (y_true == c).astype(int)
            scores = y_proba[:, ci]

            perf_auc[i, ci] = roc_auc_score(y_bin, scores)
            perf_aup[i, ci] = average_precision_score(y_bin, scores)

            fpr, tpr, _ = roc_curve(y_bin, scores)
            fold_rocs.append((fpr, tpr))

            prec, rec, _ = precision_recall_curve(y_bin, scores)
            fold_prcs.append((prec, rec))

        rocs.append(fold_rocs)
        prcs.append(fold_prcs)

    if verbose:
        print("AUC mean:", perf_auc.mean(axis=0))
        print("AUC std: ", perf_auc.std(axis=0))
        print("AUP mean:", perf_aup.mean(axis=0))
        print("AUP std: ", perf_aup.std(axis=0))

    joblib.dump(perf_auc, metrics_prefix + "_AUC.pkl")
    joblib.dump(perf_aup, metrics_prefix + "_AUP.pkl")
    joblib.dump(rocs,     metrics_prefix + "_ROC.pkl")
    joblib.dump(prcs,     metrics_prefix + "_PRC.pkl")
    joblib.dump(models, models_prefix + "_models.pkl")

    return perf_auc, perf_aup, models

