"""
Models module.

"""

import os
import xgboost as xgb

SEED = 108
CV_RESULT_DIR = "./output/xgboost_cv_results"


def OptunaObjective(trial, dtrain: xgb.DMatrix, folds: list, n_boost_round: int):
    """
    Optuna objective function definition.

    """

    param = {
        "verbosity": 0,
        "n_jobs": -1,
        "tree_method": "gpu_hist",
        "objective": "binary:logistic",
        "eval_metric": "error",
        "lambda": trial.suggest_float("lambda", 0.25, 1, log=True),
        "alpha": trial.suggest_float("alpha", 1e-9, 0.01, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.25, 1.0),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        # minimum child weight, larger the term more conservative the tree.
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "gamma": trial.suggest_float("gamma", 1e-9, 0.01, log=True),
    }

    xgb_cv_results = xgb.cv(
        params=param,
        dtrain=dtrain,
        num_boost_round=n_boost_round,
        nfold=len(folds),
        folds=folds,
        early_stopping_rounds=25,
        seed=SEED,
        verbose_eval=False,
    )

    # Set n_estimators as a trial attribute; Accessible via study.trials_dataframe().
    trial.set_user_attr("n_estimators", len(xgb_cv_results))

    # Save cross-validation results.
    filepath = os.path.join(CV_RESULT_DIR, "{}.csv".format(trial.number))
    xgb_cv_results.to_csv(filepath, index=False)

    # Extract the best score.
    best_score_max = xgb_cv_results["test-error-mean"].values[-1]

    return best_score_max
