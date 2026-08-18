import pickle

import matplotlib.pyplot as plt
import pandas as pd
from colorama import Fore
from IPython.display import display
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from public.log_utils import log_time

LOG_REG = "logistic_regression"
RAN_FOR = "random_forest"
DUMMY = "dummy_classifier"
DEC_TR = "decision_tree"
KNN = "k_nearest_neighbors"
EXTRA = "extra_trees"
XGB = "extreme_gradient_boosting"
LGBM = "light_gradient_boosting_machine"


class ClassificationResult:
    def __init__(
        self,
        title,
        classes,
        pr_auc,
        roc_auc,
        accuracy,
        classification_report,
        confusion_matrix,
    ):
        self.title = title
        self.classes = classes
        self.pr_auc = pr_auc
        self.roc_auc = roc_auc
        self.accuracy = accuracy
        self.classification_report = classification_report
        self.confusion_matrix = confusion_matrix


@log_time
def dummy_classifier(df: pd.DataFrame, target_column: str, opts: dict):
    X, y = df.drop(columns=[target_column]), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=opts.get("test_size", 0.2), random_state=42, stratify=y
    )
    model = DummyClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    return ClassificationResult(
        title=opts.get("alter_title", DUMMY),
        classes=model.classes_,
        pr_auc=average_precision_score(y_test, y_prob),
        roc_auc=roc_auc_score(y_test, y_prob, multi_class="ovr"),
        accuracy=accuracy_score(y_test, y_pred),
        classification_report=classification_report(
            y_test, y_pred, target_names=model.classes_, output_dict=True
        ),
        confusion_matrix=confusion_matrix(y_test, y_pred),
    )


@log_time
def logistic_regression(df: pd.DataFrame, target_column: str, opts: dict):
    X, y = df.drop(columns=[target_column]), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=opts.get("test_size", 0.2), random_state=42, stratify=y
    )
    model = LogisticRegression(
        C=opts.get("C", 1),
        solver=opts.get("solver", "lbfgs"),
        max_iter=opts.get("max_iter", 100),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    return ClassificationResult(
        title=opts.get("alter_title", LOG_REG),
        classes=model.classes_,
        pr_auc=average_precision_score(y_test, y_prob),
        roc_auc=roc_auc_score(y_test, y_prob, multi_class="ovr"),
        accuracy=accuracy_score(y_test, y_pred),
        classification_report=classification_report(
            y_test, y_pred, target_names=model.classes_, output_dict=True
        ),
        confusion_matrix=confusion_matrix(y_test, y_pred),
    )


@log_time
def extra_trees(df: pd.DataFrame, target_column: str, opts: dict):
    X, y = df.drop(columns=[target_column]), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=opts.get("test_size", 0.2), random_state=42, stratify=y
    )
    model = ExtraTreesClassifier(
        n_estimators=opts.get("n_estimators", 100),
        max_depth=opts.get("max_depth", None),
        min_samples_leaf=opts.get("min_samples_leaf", 1),
        criterion=opts.get("criterion", "gini"),
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    return ClassificationResult(
        title=opts.get("alter_title", EXTRA),
        classes=model.classes_,
        pr_auc=average_precision_score(y_test, y_prob),
        roc_auc=roc_auc_score(y_test, y_prob, multi_class="ovr"),
        accuracy=accuracy_score(y_test, y_pred),
        classification_report=classification_report(
            y_test, y_pred, target_names=model.classes_, output_dict=True
        ),
        confusion_matrix=confusion_matrix(y_test, y_pred),
    )


@log_time
def decision_tree(df: pd.DataFrame, target_column: str, opts: dict):
    X, y = df.drop(columns=[target_column]), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=opts.get("test_size", 0.2), random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(
        criterion=opts.get("criterion", "gini"),
        max_depth=opts.get("max_depth", None),
        min_samples_leaf=opts.get("min_samples_leaf", 1),
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    return ClassificationResult(
        title=opts.get("alter_title", DEC_TR),
        classes=model.classes_,
        pr_auc=average_precision_score(y_test, y_prob),
        roc_auc=roc_auc_score(y_test, y_prob, multi_class="ovr"),
        accuracy=accuracy_score(y_test, y_pred),
        classification_report=classification_report(
            y_test, y_pred, target_names=model.classes_, output_dict=True
        ),
        confusion_matrix=confusion_matrix(y_test, y_pred),
    )


@log_time
def random_forest(df: pd.DataFrame, target_column: str, opts: dict):
    X, y = df.drop(columns=[target_column]), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=opts.get("test_size", 0.2), random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=opts.get("n_estimators", 100),
        min_samples_leaf=opts.get("min_samples_leaf", 1),
        max_depth=opts.get("max_depth", None),
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    return ClassificationResult(
        title=opts.get("alter_title", RAN_FOR),
        classes=model.classes_,
        pr_auc=average_precision_score(y_test, y_prob),
        roc_auc=roc_auc_score(y_test, y_prob, multi_class="ovr"),
        accuracy=accuracy_score(y_test, y_pred),
        classification_report=classification_report(
            y_test, y_pred, target_names=model.classes_, output_dict=True
        ),
        confusion_matrix=confusion_matrix(y_test, y_pred),
    )


@log_time
def k_nearest_neighbors(df: pd.DataFrame, target_column: str, opts: dict):
    X, y = df.drop(columns=[target_column]), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=opts.get("test_size", 0.2), random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(
        n_neighbors=opts.get("n_neighbors", 5),
        weights=opts.get("weights", "uniform"),
        metric=opts.get("metric", "minkowski"),
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    return ClassificationResult(
        title=opts.get("alter_title", KNN),
        classes=model.classes_,
        pr_auc=average_precision_score(y_test, y_prob),
        roc_auc=roc_auc_score(y_test, y_prob),
        accuracy=accuracy_score(y_test, y_pred),
        classification_report=classification_report(
            y_test, y_pred, target_names=model.classes_, output_dict=True
        ),
        confusion_matrix=confusion_matrix(y_test, y_pred),
    )


@log_time
def extreme_gradient_boosting(df: pd.DataFrame, target_column: str, opts: dict):
    X, y = df.drop(columns=[target_column]), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=opts.get("test_size", 0.2), random_state=42, stratify=y
    )
    model = XGBClassifier(
        n_estimators=opts.get("n_estimators", 100),  # Number of gradient boosted trees
        max_depth=opts.get("max_depth", 6),  # Maximum tree depth for base learners
        learning_rate=opts.get(
            "learning_rate", 0.3
        ),  # Boosting learning rate (shrinkage)
        objective="binary:logistic",  # Loss function to optimize
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    return ClassificationResult(
        title=opts.get("alter_title", XGB),
        classes=model.classes_,
        pr_auc=average_precision_score(y_test, y_prob),
        roc_auc=roc_auc_score(y_test, y_prob, multi_class="ovr"),
        accuracy=accuracy_score(y_test, y_pred),
        classification_report=classification_report(
            y_test, y_pred, target_names=model.classes_, output_dict=True
        ),
        confusion_matrix=confusion_matrix(y_test, y_pred),
    )


def light_gradient_boosting_machine(df: pd.DataFrame, target_column: str, opts: dict):
    X, y = df.drop(columns=[target_column]), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=opts.get("test_size", 0.2), random_state=42, stratify=y
    )
    model = LGBMClassifier(
        n_estimators=opts.get("n_estimators", 100),
        learning_rate=opts.get("learning_rate", 0.1),
        num_leaves=opts.get("num_leaves", 31),
        max_depth=opts.get("max_depth", -1),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    return ClassificationResult(
        title=opts.get("alter_title", LGBM),
        classes=model.classes_,
        pr_auc=average_precision_score(y_test, y_prob),
        roc_auc=roc_auc_score(y_test, y_prob, multi_class="ovr"),
        accuracy=accuracy_score(y_test, y_pred),
        classification_report=classification_report(
            y_test, y_pred, target_names=model.classes_, output_dict=True
        ),
        confusion_matrix=confusion_matrix(y_test, y_pred),
    )


def report(model: ClassificationResult, matrix: bool = False) -> None:
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.float_format", "{:.3f}".format)

    print(f"\n\t{Fore.GREEN}{model.title}{Fore.RESET}")
    display(
        pd.DataFrame(
            [
                {
                    "pr_auc": model.pr_auc,
                    "roc_auc": model.roc_auc,
                    "accuracy": model.accuracy,
                }
            ]
        ).style.hide(axis="index")
    )
    display(pd.DataFrame(model.classification_report).transpose())
    if matrix:
        ConfusionMatrixDisplay(
            confusion_matrix=model.confusion_matrix,
            display_labels=model.classes,
        ).plot(cmap=plt.cm.Blues)
        plt.show()


def pick(data, name: str) -> None:
    with open(f"./data/pick/{name}.pkl", "wb") as f:
        pickle.dump(data, f)
