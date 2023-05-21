"""
Module for Explainable Artificial Intelligence.

"""

import optuna

import numpy as np
import pandas as pd
import xgboost as xgb

from src.utils import reduceMemory
from src.models import OptunaObjective


class ExplainSumm:
    """
    XAI class.
    
    """

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: list,
        non_features: list,
        n_trials: int = 50,
        n_boost_round: int = 500,
    ):
        """
        Initialize the ExplainSumm class.

        """
        self.train = reduceMemory(train)
        self.test = reduceMemory(test)
        self.target = target
        self.non_features = non_features
        self.n_trials = n_trials
        self.n_boost_round: int = n_boost_round

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.customFolds = None
        self.xgtrain = None
        self.xgtest = None

        self.optuna_study = None
        self.best_params = None

        self.model = None
        self.test_pred = None

        self.main()

    def main(self):
        """
        Main execution.

        """

        # Generate comparisons features
        self.generate_spread_drift_features()

        # Generate batch folds
        self.batch_folds()

        # Custom CV train
        self.xgb_optuna_batch_CV()

        # Predict
        self.predict()

    def generate_spread_drift_features(self):
        """
        Generate comparisons features. Spread and drift.

        """

        # Base features
        features = self.train.columns[~self.train.columns.isin(self.non_features)]
        base_features = set(map(lambda x: "_".join(x.split("_")[1:]), features))

        # Train-Test
        self.X_train, self.y_train = (
            self.features_spread_drift(self.train, base_features),
            self.train[self.target],
        )
        self.X_test, self.y_test = (
            self.features_spread_drift(self.test, base_features),
            self.test[self.target],
        )

    @staticmethod
    def features_spread_drift(data: pd.DataFrame, base_features: set) -> pd.DataFrame:
        """
        Calculate the spread and drift for each base feature in the given data.

        Args:
            data (pd.DataFrame): The input data containing the columns to 
            compute spread and drift for.
            base_features (list): A list of base feature names to calculate 
            spread and drift for.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated spread and 
            drift values for each base feature.
            
        """

        # Iterate through the base features
        for i, feature in enumerate(base_features):

            # Base features
            a_feature = f"m0_{feature}"
            b_feature = f"m1_{feature}"

            # Calculate the spread and drift for the current base feature
            spread_drift = pd.concat(
                [
                    data[a_feature]
                    .subtract(data[b_feature])
                    .rename(f"spread_{feature}"),
                    data[a_feature]
                    .add(data[b_feature])
                    .div(2)
                    .rename(f"drift_{feature}"),
                ],
                axis=1,
            )

            if i == 0:
                df = spread_drift
            else:
                df = pd.concat([df, spread_drift], axis=1)

        return df

    def batch_folds(self):
        """
        Create custom folds attending to the batch features.

        """

        self.customFolds = []

        # Get unique batchs
        unique_batchs = self.train["batch"].unique()

        # Train, test index batch kfolds
        for batch in unique_batchs:
            self.customFolds.append(
                (
                    self.train[self.train["batch"] != batch].index.to_list(),
                    self.train[self.train["batch"] == batch].index.to_list(),
                )
            )

    def xgb_optuna_batch_CV(self):
        """
        XGB classifier with Optuna and batch CV.

        """

        # XGB train and test set
        self.xgtrain = xgb.DMatrix(
            self.X_train.values, self.y_train.values, feature_names=self.X_train.columns
        )
        self.xgtest = xgb.DMatrix(
            self.X_test.values, self.y_test.values, feature_names=self.X_test.columns
        )

        # OptunaStudy
        self.optuna_study = optuna.create_study(direction="minimize")
        self.optuna_study.optimize(
            lambda trial: OptunaObjective(
                trial, self.xgtrain, self.customFolds, self.n_boost_round
            ),
            n_trials=self.n_trials,
        )

        # Best model
        self.best_params = pd.DataFrame(
            map(
                lambda x: [x.number, x.values[0], x.params, x.user_attrs],
                self.optuna_study.best_trials,
            ),
            columns=["number", "test-error-mean", "params", "user_attrs"],
        )

    def predict(self):
        """
        Once Optuna has found out the best params combination, predict test 
        data and calculate accuracy and related metrics.

        """

        # Training
        print("Training model...")
        self.model = xgb.train(
            self.best_params.loc[0, "params"],
            self.xgtrain,
            num_boost_round=self.best_params.loc[0, "user_attrs"]["n_estimators"],
        )

        # Generate predictions
        print("Generating predictions...")
        self.test_pred = self.model.predict(self.xgtest)
