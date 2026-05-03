import numpy as np

def classify_db(
    db,
    model_per_fold,
    nut_sel,
    expected_classes=None,
    fpro_low_class=None,
    fpro_high_class=None,
    copy=True
):
    """
    Classify a database using one fitted scikit-learn classifier per fold (version independent of number of folds and classes).
    It checks that:
    - all models are fitted;
    - all models expose predict(), predict_proba(), and classes_;
    - all models have the same class universe and class order;
    - all models have seen all expected classes;
    - feature number/order is consistent when the model stores feature metadata.

    Parameters
    ----------
    db : pandas.DataFrame
        Input dataframe.

    model_per_fold : list
        List of fitted scikit-learn classifiers.

    nut_sel : list
        Ordered list of feature columns used for prediction.

    expected_classes : list, optional
        Full intended class universe.
        Example: [0, 1, 2, 3] or [1, 2, 3, 4].
        If provided, each model must have exactly these classes.
        If not provided, the first model's classes_ are used as reference.

    fpro_low_class : optional
        Class corresponding to the least processed endpoint.
        If None, defaults to the first class in expected_classes.

    fpro_high_class : optional
        Class corresponding to the most processed endpoint.
        If None, defaults to the last class in expected_classes.

    copy : bool
        If True, returns a modified copy of db.
        If False, modifies db in place.

    Returns
    -------
    pandas.DataFrame
        Dataframe with fold-level probabilities, mean probabilities,
        FPro, std_FPro, min_FPro, class calls.
    """

    if len(model_per_fold) == 0:
        raise ValueError("model_per_fold is empty.")

    if copy:
        db = db.copy()

    nut_sel = list(nut_sel)

    missing_features = [col for col in nut_sel if col not in db.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in db: {missing_features}")

    Xnut = db.loc[:, nut_sel].to_numpy()
    n_samples = db.shape[0]
    n_folds = len(model_per_fold)

    first_model = model_per_fold[0]

    if not hasattr(first_model, "classes_"):
        raise TypeError("The first model does not expose classes_. Is it fitted?")

    if expected_classes is None:
        expected_classes = np.array(first_model.classes_)
    else:
        expected_classes = np.array(expected_classes)

    n_classes = len(expected_classes)

    if n_classes < 2:
        raise ValueError("At least two classes are required.")

    if fpro_low_class is None:
        fpro_low_class = expected_classes[0]

    if fpro_high_class is None:
        fpro_high_class = expected_classes[-1]

    if fpro_low_class not in expected_classes:
        raise ValueError(
            f"fpro_low_class={fpro_low_class} is not in expected_classes={expected_classes}."
        )

    if fpro_high_class not in expected_classes:
        raise ValueError(
            f"fpro_high_class={fpro_high_class} is not in expected_classes={expected_classes}."
        )

    low_idx = int(np.where(expected_classes == fpro_low_class)[0][0])
    high_idx = int(np.where(expected_classes == fpro_high_class)[0][0])


    # Validate all models

    for fold_id, model in enumerate(model_per_fold, start=1):

        if not hasattr(model, "predict"):
            raise TypeError(f"Model fold {fold_id} does not implement predict().")

        if not hasattr(model, "predict_proba"):
            raise TypeError(f"Model fold {fold_id} does not implement predict_proba().")

        if not hasattr(model, "classes_"):
            raise TypeError(f"Model fold {fold_id} does not expose classes_. Is it fitted?")

        if not np.array_equal(model.classes_, expected_classes):
            raise ValueError(
                f"Model fold {fold_id} has classes {model.classes_}, "
                f"but expected classes are {expected_classes}.\n"
                "This means the model did not see all expected classes, "
                "or the class order is inconsistent."
            )

        if hasattr(model, "n_features_in_"):
            if model.n_features_in_ != len(nut_sel):
                raise ValueError(
                    f"Model fold {fold_id} expects {model.n_features_in_} features, "
                    f"but nut_sel has {len(nut_sel)} features."
                )

        if hasattr(model, "feature_names_in_"):
            model_feature_names = list(model.feature_names_in_)
            if model_feature_names != nut_sel:
                raise ValueError(
                    f"Feature names/order mismatch in fold {fold_id}.\n"
                    f"Model feature_names_in_: {model_feature_names}\n"
                    f"Provided nut_sel:        {nut_sel}"
                )


    # Predict with all models

    prob_by_fold = []
    pred_by_fold = []
    fpro_by_fold = []

    for fold_id, model in enumerate(model_per_fold, start=1):

        y_pred = model.predict(Xnut)
        y_probs = model.predict_proba(Xnut)

        expected_shape = (n_samples, n_classes)

        if y_probs.shape != expected_shape:
            raise ValueError(
                f"Fold {fold_id} predict_proba returned shape {y_probs.shape}, "
                f"but expected {expected_shape}."
            )

        db[f"classf{fold_id}"] = y_pred

        for class_idx in range(n_classes):
            db[f"p{class_idx + 1}f{fold_id}"] = y_probs[:, class_idx]

        fpro_fold = (1 - y_probs[:, low_idx] + y_probs[:, high_idx]) / 2
        db[f"FProf{fold_id}"] = fpro_fold

        prob_by_fold.append(y_probs)
        pred_by_fold.append(y_pred)
        fpro_by_fold.append(fpro_fold)

    # samples x classes x folds
    probs = np.stack(prob_by_fold, axis=2)

    # samples x folds
    preds = np.column_stack(pred_by_fold)
    fpros = np.column_stack(fpro_by_fold)

    ddof = 1 if n_folds > 1 else 0

   
    # Average class probabilities across folds

    mean_probs = probs.mean(axis=2)

    for class_idx in range(n_classes):
        p_col = f"p{class_idx + 1}"
        db[p_col] = mean_probs[:, class_idx]
        db[f"std_{p_col}"] = probs[:, class_idx, :].std(axis=1, ddof=ddof)

    # Average FPro across folds

    db["FPro"] = fpros.mean(axis=1)
    db["std_FPro"] = fpros.std(axis=1, ddof=ddof)
    db["min_FPro"] = fpros.min(axis=1)
    db["max_FPro"] = fpros.max(axis=1)

    # Final class from averaged probabilities

    max_class_idx = np.argmax(mean_probs, axis=1)

    db["max_p"] = [f"p{i + 1}" for i in max_class_idx]
    db["class"] = expected_classes[max_class_idx]

    # Fold with minimum FPro

    min_fold_idx = np.argmin(fpros, axis=1)

    db["min_in_which_fold"] = [f"FProf{i + 1}" for i in min_fold_idx]
    db["min_fold_id"] = min_fold_idx + 1

    row_idx = np.arange(n_samples)

    db["min_class"] = preds[row_idx, min_fold_idx]

    for class_idx in range(n_classes):
        db[f"p{class_idx + 1}_minFPro"] = probs[row_idx, class_idx, min_fold_idx]


    return db