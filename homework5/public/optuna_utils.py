import catboost as cb
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import (
    KFold,
    cross_val_score,
)
from xgboost import XGBRegressor

OPT_RAN_FOR = "Random Forest"
OPT_XGB_SL = "Extreme Gradient Boosting (sklearn)"
OPT_XGB = "Extreme Gradient Boosting"
OPT_LGBM = "Light Gradient-Boosting Machine"
OPT_CAT = "Categorical boosting"
ALLOWED_MODEL_TYPE = [OPT_RAN_FOR, OPT_XGB_SL, OPT_XGB, OPT_LGBM, OPT_CAT]


def optimize(
    model_type: str,
    df: pd.DataFrame,
    target_column: str,
    n_trials: int = 20,
    max_iter: int = 100,
    # Complete silence (suppresses warnings too)
    log_level: int = optuna.logging.ERROR,
) -> optuna.Study:
    optuna.logging.set_verbosity(log_level)

    X, y = df.drop(columns=[target_column]), df[target_column]
    study = optuna.create_study(
        study_name=model_type,
        direction="maximize",
    )

    # @log_time
    def optuna_optimize(trial):
        if model_type == OPT_RAN_FOR:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 1.0]
                ),
            }
            model = RandomForestRegressor(**params)
            cv = KFold(n_splits=5, shuffle=False)
            return cross_val_score(model, X, y, cv=cv, scoring="r2").mean()
        elif model_type == OPT_XGB:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                # 'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                # 'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                # 'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            }
            model = XGBRegressor(**params)
            cv = KFold(n_splits=5, shuffle=False)
            return cross_val_score(model, X, y, cv=cv, scoring="r2").mean()
        elif model_type == OPT_XGB_SL:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_float("max_features", 0.1, 1.0),
            }
            model = ExtraTreesRegressor(**params)
            cv = KFold(n_splits=5, shuffle=False)
            return cross_val_score(model, X, y, cv=cv, scoring="r2").mean()
        elif model_type == OPT_LGBM:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                "verbose": -1,
            }
            model = LGBMRegressor(**params)
            cv = KFold(n_splits=5, shuffle=False)
            return cross_val_score(model, X, y, cv=cv, scoring="r2").mean()
        elif model_type == OPT_CAT:
            params = {
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "depth": trial.suggest_int("depth", 3, 9),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                "loss_function": "RMSE",
                "eval_metric": "R2",
                "logging_level": "Silent",
            }
            cv_dataset = cb.Pool(data=X, label=y)
            cv_results = cb.cv(
                cv_dataset,
                params,
                fold_count=5,
                partition_random_seed=42,
                return_models=False,
                early_stopping_rounds=50,
                verbose=False,
                shuffle=False,
            )
            return np.max(cv_results["test-R2-mean"])
        return None

    study.optimize(
        optuna_optimize, n_trials=n_trials, n_jobs=-1, show_progress_bar=True, catch=()
    )
    return study
