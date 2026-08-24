from typing import Self

import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from optuna import Study
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    OrthogonalMatchingPursuit,
    Ridge,
    SGDRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import public.regression_utils as reg
from public.optuna_utils import (
    OPT_CAT,
    OPT_LGBM,
    OPT_RAN_FOR,
    OPT_XGB,
    OPT_XGB_SL,
    optimize,
)

LIN_REG = "linear_regression"
RAN_FOR = "random_forest"
LGBM = "light_gradient_boosting_machine"
G_BOOST = "gradient_boosting"
XGB = "extreme_gradient_boosting"
XBT_SL = "extreme_gradient_boosting (sklearn)"
CAT_BST = "categorical_boosting"
ADA_BST = "adaptive_boosting"
LASSO = "lasso_l1_regularization"
RIDGE = "ridge_l2_regularization"
BAYES = "bayesian_ridge_regression"
HUBER = "huber_regressor"
DEC_TR = "decision_tree_regressor"
OMP = "orthogonal_matching_pursuit"
PASS_AGR = "passive_aggressive_regressor"
KNN = "k_neighbors_regressor"
EL_NET = "elastic_net"


MODE_DEFAULT = 1
MODE_OPTUNA = 2
MODE_BEST = 3
MODE_ALL = 4


OPTUNA_DICT = {
    XGB: OPT_XGB,
    XBT_SL: OPT_XGB_SL,
    LGBM: OPT_LGBM,
    RAN_FOR: OPT_RAN_FOR,
    CAT_BST: OPT_CAT,
}


class ModelNotFoundException(Exception):
    pass


class RegressionProcessor:
    def __init__(self, df: pd.DataFrame, target_column: str):
        # self.original = df
        self.target_column = target_column
        self.models = []
        self.studies = []
        self.best_models = []
        X, y = df.drop(columns=[target_column]), df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.train_df = pd.concat([X_train, y_train], axis=1)
        self.test_df = pd.concat([X_test, y_test], axis=1)

    def calculate(self, algorithm_list: dict[str, dict], mode=MODE_DEFAULT) -> Self:
        """
        train + predict + evaluate + prepare data for report
        """
        if mode == MODE_DEFAULT:
            for key in algorithm_list:
                opts = algorithm_list.get(key)
                self.models.append(
                    reg.regression(
                        model=self._get_model(key),
                        df=self.train_df.copy(),
                        target_column=self.target_column,
                        title=key,
                        opts=opts,
                    )
                )
        elif mode == MODE_OPTUNA:
            for key in algorithm_list:
                self.studies.append(
                    {
                        "title": key,
                        "study": optimize(
                            model_type=OPTUNA_DICT[key],
                            df=self.train_df.copy(),
                            target_column=self.target_column,
                            n_trials=50,
                        ),
                    }
                )
        elif mode == MODE_BEST:
            for key in algorithm_list:
                opts = algorithm_list.get(key)
                opts.update(self._study_by_key(key)["study"].best_params)
                self.best_models.append(
                    reg.regression(
                        model=self._get_model(key),
                        df=self.train_df.copy(),
                        target_column=self.target_column,
                        title=key,
                        opts=opts,
                    )
                )
        return self

    def _study_by_key(self, key: str) -> Study:
        return next((study for study in self.studies if study["title"] == key), None)

    def report(self, title: str) -> None:
        if title.endswith("_tuned"):
            model = next(
                (item for item in self.best_models if item.title == title[:-6]), None
            )
            if not model:
                raise ModelNotFoundException(f"Best model for '{title}' not found")
            reg.report(self.test_df, self.target_column, model)

            study = next(
                (item["study"] for item in self.studies if item["title"] == title[:-6]),
                None,
            )
            if not study:
                raise ModelNotFoundException(f"Optuna info for '{title}' not found")
            reg.optuna_plots(study)
        else:
            model = next((item for item in self.models if item.title == title), None)
            if not model:
                raise ModelNotFoundException(f"Model for '{title}' not found")
            reg.report(self.test_df, self.target_column, model)

    def summary_table(self, titles: list[str], mode=MODE_DEFAULT) -> str | None:
        if mode == MODE_DEFAULT:
            result = [model for model in self.models if model.title in titles]
            reg.summary_report(result.copy())
        elif mode == MODE_OPTUNA:
            reg.optuna_report(self.studies)
        elif mode == MODE_BEST:
            result = [model for model in self.best_models if model.title in titles]
            reg.summary_report(result.copy())
        elif mode == MODE_ALL:
            return reg.super_report(self.models, self.best_models)

    def pick_model(self, title: str) -> None:
        model = next((item for item in self.models if item.title == title), None)
        if not model:
            raise ModelNotFoundException(f"Model for '{title}' not found")
        reg.pick(model, title)

    def _get_model(self, name: str):
        if name == LIN_REG:
            return LinearRegression()
        elif name == RAN_FOR:
            return RandomForestRegressor()
        elif name == LGBM:
            return LGBMRegressor()
        elif name == G_BOOST:
            return GradientBoostingRegressor()
        elif name == XGB:
            return XGBRegressor()
        elif name == XBT_SL:
            return ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif name == CAT_BST:
            return CatBoostRegressor(
                iterations=500, learning_rate=0.05, depth=6, random_state=42, verbose=0
            )
        elif name == ADA_BST:
            return AdaBoostRegressor(n_estimators=100, random_state=42)
        elif name == LASSO:
            return Lasso(alpha=0.1)
        elif name == RIDGE:
            return Ridge(alpha=1.0, random_state=42)
        elif name == BAYES:
            return BayesianRidge()
        elif name == HUBER:
            return HuberRegressor(epsilon=100)
        elif name == DEC_TR:
            return DecisionTreeRegressor(max_depth=15)
        elif name == OMP:
            return OrthogonalMatchingPursuit(n_nonzero_coefs=5)
        elif name == PASS_AGR:
            return SGDRegressor(
                loss="epsilon_insensitive", penalty=None, learning_rate="pa1", eta0=1.0
            )
        elif name == KNN:
            return KNeighborsRegressor(n_neighbors=11)
        elif name == EL_NET:
            return ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        else:
            raise ModelNotFoundException(f"Algorithm {name} not supported yet")
