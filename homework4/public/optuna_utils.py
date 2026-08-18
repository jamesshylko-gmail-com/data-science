import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    make_scorer,
    precision_recall_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier

from public.log_utils import log_time

OPT_LOG_REG = "Logistic Regression"
OPT_RAN_FOR = "Random Forest"
OPT_KNN = "K-nearest neighbours"
OPT_EXTRA = "Extra Trees"
OPT_DEC_TR = "Decision Tree"
OPT_XGB = "Extreme Gradient Boosting"
OPT_LGBM = "Light Gradient-Boosting Machine"
ALLOWED_MODEL_TYPE = [
    OPT_LOG_REG,
    OPT_RAN_FOR,
    OPT_KNN,
    OPT_EXTRA,
    OPT_DEC_TR,
    OPT_XGB,
    OPT_LGBM,
]


def optimize(
    model_type: str,
    df: pd.DataFrame,
    target_column: str,
    n_trials: int = 20,
    max_iter: int = 100,
    # Complete silence (suppresses warnings too)
    log_level: int = optuna.logging.INFO,
) -> optuna.Study:
    optuna.logging.set_verbosity(log_level)

    X, y = df.drop(columns=[target_column]), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # cоздаем объект исследования
    study = optuna.create_study(
        study_name=model_type,
        direction="maximize",
    )

    @log_time
    def optuna_optimize(trial):
        # задаем пространства поиска гиперпараметров
        if model_type == OPT_LOG_REG:
            model = LogisticRegression(
                l1_ratio=0,  # l2
                solver=trial.suggest_categorical("solver", ["lbfgs", "sag"]),
                C=trial.suggest_float("C", 0.1, 2, log=True),
                random_state=42,
                class_weight="balanced",
                max_iter=trial.suggest_int(name="max_iter", low=100, high=200, step=10),
            )

        elif model_type == OPT_RAN_FOR:
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int(
                    name="n_estimators", low=100, high=200, step=10
                ),
                max_depth=trial.suggest_int(name="max_depth", low=3, high=10, step=1),
                min_samples_leaf=trial.suggest_int(
                    name="min_samples_leaf", low=3, high=7, step=1
                ),
                random_state=42,
                class_weight="balanced",
            )
        elif model_type == OPT_KNN:
            model = KNeighborsClassifier(
                n_neighbors=trial.suggest_int(
                    name="n_neighbors", low=1, high=30, step=1
                ),
                weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                metric=trial.suggest_categorical(
                    "metric", ["euclidean", "manhattan", "minkowski"]
                ),
            )
        elif model_type == OPT_EXTRA:
            params = {
                "n_estimators": trial.suggest_int(
                    name="n_estimators", low=50, high=500, step=50
                ),
                "max_depth": trial.suggest_int(name="max_depth", low=3, high=30),
                # "min_samples_split": trial.suggest_int(name="min_samples_split", low=2, high=20),
                "min_samples_leaf": trial.suggest_int(
                    name="min_samples_leaf", low=1, high=20
                ),
                # "max_features": trial.suggest_float("max_features", 0.1, 1.0),
                "criterion": trial.suggest_categorical(
                    "criterion", ["gini", "entropy"]
                ),
                # "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "random_state": 42,
                # "n_jobs": -1
            }
            model = ExtraTreesClassifier(**params)
        elif model_type == OPT_DEC_TR:
            params = {
                "criterion": trial.suggest_categorical(
                    "criterion", ["gini", "entropy"]
                ),
                # "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
                "max_depth": trial.suggest_int(name="max_depth", low=2, high=32),
                # "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int(
                    name="min_samples_leaf", low=1, high=20
                ),
                "random_state": 42,
            }
            model = DecisionTreeClassifier(**params)
        elif model_type == OPT_XGB:
            # Функция для расчета multiclass PR-AUC (One-vs-Rest)
            def multiclass_pr_auc_score(y_true, y_proba):
                y_true_bin = label_binarize(y_true, classes=list(range(10)))
                return average_precision_score(y_true_bin, y_proba, average="macro")

            # Создаем скорер для cross_validate
            pr_auc_scorer = make_scorer(
                multiclass_pr_auc_score, response_method="predict_proba"
            )

            # Инициализируем модель Scikit-Learn API
            model = xgb.XGBClassifier(
                n_estimators=trial.suggest_int(name="n_estimators", low=50, high=100),
                max_depth=trial.suggest_int(name="max_depth", low=3, high=5),
                learning_rate=trial.suggest_float("learning_rate", 0.1, 0.5, log=True),
                objective="multi:softprob",
                num_class=10,
                eval_metric="mlogloss",  # Внутренняя метрика для обучения
            )
            cv_results = cross_validate(
                model,
                X,
                y,
                cv=5,
                scoring={"pr_auc": pr_auc_scorer},
                return_train_score=False,
            )
            # Оптимизируемое значение для Optuna (максимизируем PR-AUC)
            return cv_results["test_pr_auc"].mean()
        elif model_type == OPT_LGBM:

            def pr_auc_metric(preds, train_data):
                """
                Кастомная метрика для LightGBM v4.x+.
                В многоклассовом режиме preds имеет форму (n_samples, n_classes).
                """
                labels = train_data.get_label()
                n_classes = 10

                # Изменяем форму, если массив плоский (зависит от версии и fobj, но в v4.x cv передает матрицу)
                if preds.ndim == 1:
                    preds = preds.reshape(-1, n_classes)

                pr_aucs = []
                for i in range(n_classes):
                    # Бинаризация меток для стратегии One-vs-Rest
                    y_true_cls = (labels == i).astype(int)
                    y_pred_cls = preds[:, i]

                    precision, recall, _ = precision_recall_curve(
                        y_true_cls, y_pred_cls
                    )
                    # Рассчитываем площадь под PR-кривой для текущего класса
                    pr_aucs.append(auc(recall, precision))

                macro_pr_auc = np.mean(pr_aucs)

                # Возвращает: (имя_метрики, значение, стремиться_ли_к_максимуму)
                return "macro_pr_auc", macro_pr_auc, True

            params = {
                "objective": "multiclass",
                "num_class": 10,
                "metric": "multi_logloss",  # or 'multi_error'
                "boosting_type": "gbdt",
                "n_estimators": trial.suggest_int("n_estimators", 50, 100),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.05, 0.1, log=True
                ),
                "num_leaves": trial.suggest_int(name="num_leaves", low=60, high=100),
                # "max_depth": trial.suggest_int(name="max_depth", low=6, high=10),
                "max_depth": -1,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "min_child_samples": 50,
                "verbose": -1,
                "random_state": 42,
            }
            train_data = lgb.Dataset(X, label=y)

            # Запуск кросс-валидации с early stopping и PR-AUC
            cv_results = lgb.cv(
                params,
                train_data,
                num_boost_round=100,
                nfold=5,
                stratified=True,
                shuffle=True,
                feval=pr_auc_metric,
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=50, first_metric_only=False, verbose=True
                    ),
                    lgb.log_evaluation(period=10),
                ],
            )
            # print(f"{Fore.LIGHTRED_EX}cv_results.items(){Fore.RESET}")
            # for key, values in cv_results.items():
            #     print(f"{key}: {values[-1]}")
            return max(cv_results["valid macro_pr_auc-mean"])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pr_auc = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="average_precision"
        ).mean()
        return pr_auc

    study.optimize(
        optuna_optimize, n_trials=n_trials, n_jobs=-1, show_progress_bar=True
    )
    return study
