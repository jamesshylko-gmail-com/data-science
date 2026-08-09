from pprint import pp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

from public.log_utils import log_time

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"


@log_time
def logistic_regression_softmax(df: pd.DataFrame, target_column: str, opts: dict):
    result = {
        "title": "Softmax regression (multinomial logistic regression)",
        "description": "lbfgs : Алгоритм Бройдена–Флетчера–Голдфарба–Шанно с ограниченной памятью",
        "stats": {},
    }
    X, y = df.drop(columns=[target_column]), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=opts.get("test_size", 0.2), random_state=42
    )

    # 'multinomial' enables the Softmax function for multi-class classification
    model = LogisticRegression(solver="lbfgs", max_iter=opts.get("max_iter", 100))
    model.fit(X_train, y_train)

    # Extract raw model outputs (logits) for the first test sample
    # Softmax uses these raw scores to calculate final probabilities
    sample_input = X_test[0:1]
    result["stats"]["Raw Logits (Scores)"] = model.decision_function(
        sample_input
    ).tolist()

    # Get the Softmax-normalized probabilities
    # This represents the actual Softmax function output
    softmax_probabilities = model.predict_proba(sample_input)
    result["stats"]["Softmax Probabilities"] = model.predict_proba(
        sample_input
    ).tolist()

    # Check that the probabilities sum up to exactly 1.0
    result["stats"]["Sum of Probabilities"] = float(np.sum(softmax_probabilities))
    result["stats"]["Predicted Class ID"] = model.predict(sample_input).tolist()

    # Получаем предсказания меток классов
    y_pred = model.predict(X_test)

    # Строим матрицу ошибок
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    result["plots"] = (disp, {"cmap": plt.cm.Reds})

    result["Classification report"] = classification_report(
        y_test, y_pred, target_names=model.classes_, output_dict=True
    )
    # Предсказание вероятностей принадлежности к классам
    y_prob = model.predict_proba(X_test)

    # Расчет потерь (Log Loss)
    # multiclass_logloss = log_loss(y_test, y_prob)
    result["stats"]["Log Loss"] = log_loss(y_test, y_prob)

    # Расчет ROC-AUC со стратегией "один против каждого" (ovo) или "один против всех" (ovr)
    result["stats"]["ROC-AUC (OvO, Macro)"] = roc_auc_score(
        y_test, y_prob, multi_class="ovo", average="macro"
    ).tolist()
    result["model"] = model
    return result


@log_time
def logistic_regression_one_vs_rest(df: pd.DataFrame, target_column: str, opts: dict):
    result = {
        "title": "Logistic Regression One-vs-Rest (OvR)",
        "description": "Стандартная бинарная логистическая регрессия, использованная для решения проблемы с тремя или более классами",
        "stats": {},
    }
    X, y = df.drop(columns=[target_column]), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=opts.get("test_size", 0.2), random_state=42
    )

    # Обертываем её в OneVsRestClassifier
    model = OneVsRestClassifier(LogisticRegression())
    # Обучаем модель (под капотом обучится 3 логистические регрессии)
    model.fit(X_train, y_train)

    # Делаем предсказания и оцениваем точность
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    result["stats"]["accuracy"] = model.score(X_test, y_test)
    result["stats"]["Количество обученных моделей"] = len(model.estimators_)

    result["Classification report"] = classification_report(
        y_test, y_pred, output_dict=True
    )

    # Оцениваем ROC-AUC (бинарная оцифровка истинных меток)
    # y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    result["stats"]["ROC-AUC (OvO, Macro)"] = roc_auc_score(
        y_test, y_prob, multi_class="ovr", average="macro"
    )

    # Вычисление матрицы ошибок nхn
    cm = confusion_matrix(y_test, y_pred)
    # Визуализация матрицы
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    result["plots"] = (disp, {"cmap": plt.cm.Oranges, "values_format": "d"})

    result["model"] = model
    return result


@log_time
def naive_bayes(df: pd.DataFrame, target_column: str, opts: dict):
    # do = ["gaussian_nb", "multinomial_nb", "bernoulli_nb", "Complement NB"]
    # Not implemented yet
    pass


@log_time
def support_vector_machine(df: pd.DataFrame, target_column: str, opts: dict):
    result = {
        "title": "Support Vector Machine, SVM",
        "description": "Основная цель SVM — найти оптимальную разделяющую гиперплоскость, которая максимизирует зазор (расстояние) между объектами разных классов.",
        "stats": {},
    }
    X, y = df.drop(columns=[target_column]), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=opts.get("test_size", 0.2), random_state=42
    )
    # Создаем модель метода опорных векторов (ядро 'rbf' используется по умолчанию)
    model = SVC(kernel="linear", C=1.0, class_weight=None, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Оцениваем качество модели
    result["stats"]["accuracy"] = accuracy_score(y_test, y_pred)
    # Количество опорных векторов: слишком большой их процент говорит о высокой сложности или зашумленности данных, малый — о стабильной границе
    result["stats"]["Общее количество опорных векторов"] = int(sum(model.n_support_))

    result["Classification report"] = classification_report(
        y_test, y_pred, target_names=model.classes_, output_dict=True
    )

    # Строим матрицу ошибок
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    result["plots"] = (disp, {"cmap": plt.cm.Greens})

    result["model"] = model
    return result


@log_time
def k_neighbors(df: pd.DataFrame, target_column: str, opts: dict):
    result = {
        "title": "Nearest Neighbors",
        "description": "Предоставляет функциональность для методов обучения на основе ближайших соседей без учителя и с учителем",
        "stats": {},
    }
    # Загрузка данных и разделение на обучающую и тестовую выборки
    X, y = df.drop(columns=[target_column]), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=opts.get("test_size", 0.2), random_state=42, stratify=y
    )

    # Настройка GridSearchCV (кросс-валидация на 5 блоков)
    # cv=5 : 80% for train, 20% for test
    model = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid={
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        cv=5,
        scoring="accuracy",
    )
    model.fit(X_train, y_train)

    # Вывод лучших параметров
    result["stats"]["Лучшие параметры"] = model.best_params_
    result["stats"]["Лучшая точность на кросс-валидации"] = float(model.best_score_)

    # Оценка результатов на тестовой выборке
    best_model = model.best_estimator_
    y_pred = best_model.predict(X_test)
    result["stats"]["Accuracy"] = accuracy_score(y_test, y_pred)

    result["Classification report"] = classification_report(
        y_test, y_pred, output_dict=True
    )

    # Строим матрицу ошибок
    result["plots"] = (
        ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(y_test, y_pred),
            display_labels=model.classes_,
        ),
        {"cmap": plt.cm.Blues},
    )
    result["model"] = best_model
    return result


@log_time
def decision_tree(df: pd.DataFrame, target_column: str, opts: dict):
    result = {
        "title": "A decision tree classifier",
        "description": "Непараметрический метод обучения с учителем, используемый для классификации и регрессии . Цель состоит в создании модели, которая предсказывает значение целевой переменной, обучаясь простым правилам принятия решений, выведенным из характеристик данных. Дерево можно рассматривать как кусочно-постоянную аппроксимацию",
        "stats": {},
    }
    # Загрузка данных
    X, y = df.drop(columns=[target_column]), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=opts.get("test_size", 0.2), random_state=42
    )

    # Создание и обучение модели
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Делаем предсказания на тесте
    y_pred = model.predict(X_test)

    # Оценка результатов
    result["stats"]["Accuracy"] = accuracy_score(y_test, y_pred)
    result["stats"]["Матрица ошибок"] = confusion_matrix(y_test, y_pred).tolist()

    result["Classification report"] = classification_report(
        y_test, y_pred, target_names=model.classes_, output_dict=True
    )

    # Визуализация дерева
    result["plots"] = {"feature_names": X.columns}

    result["model"] = model
    return result


@log_time
def random_forest(df: pd.DataFrame, target_column: str, opts: dict):
    result = {
        "title": "A random forest classifier",
        "description": "Случайный лес — это мета-оценщик, который обучает ряд классификаторов на основе деревьев решений на различных подвыборках набора данных и использует усреднение для повышения точности прогнозирования и контроля переобучения",
        "stats": {},
    }
    X, y = df.drop(columns=[target_column]), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=opts.get("test_size", 0.2), random_state=42
    )

    # Создаем модель случайного леса
    model = RandomForestClassifier(
        n_estimators=opts.get("n_estimators", 100), oob_score=True, random_state=42
    )

    model.fit(X_train, y_train)

    # Делаем предсказание и оцениваем точность
    y_pred = model.predict(X_test)
    result["stats"]["Accuracy"] = accuracy_score(y_test, y_pred)
    # Включается параметром oob_score=True при инициализации модели. Позволяет оценить качество леса на данных, которые не участвовали в построении конкретного дерева
    result["stats"]["OOB Score"] = model.oob_score_

    result["Classification report"] = classification_report(
        y_test, y_pred, target_names=model.classes_, output_dict=True
    )

    result["stats"]["Матрица ошибок"] = confusion_matrix(y_test, y_pred).tolist()

    # Важность признаков отражаемна круговой диаграмме
    result["plots"] = {
        name: float(imp) for name, imp in zip(X.columns, model.feature_importances_)
    }

    result["model"] = model
    return result


@log_time
def gradient_boosting(df: pd.DataFrame, target_column: str, opts: dict):
    result = {
        "title": "Gradient Boosted Decision Trees - GBDT",
        "description": "обобщение бустинга до произвольных дифференцируемых функций потерь, что бы это ни значило",
        "stats": {},
    }
    X, y = df.drop(columns=[target_column]), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=opts.get("test_size", 0.2), random_state=42
    )

    # Создание и обучение модели
    model = GradientBoostingClassifier(
        n_estimators=opts.get("n_estimators", 100),
        learning_rate=opts.get("learning_rate", 0.1),
        max_depth=opts.get("max_depth", 3),
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Оценка качества
    result["stats"]["Accuracy"] = accuracy_score(y_test, y_pred)
    result["stats"]["Матрица ошибок"] = confusion_matrix(y_test, y_pred).tolist()
    result["Classification report"] = classification_report(
        y_test, y_pred, target_names=model.classes_, output_dict=True
    )

    result["plots"] = {
        name: float(imp) for name, imp in zip(X.columns, model.feature_importances_)
    }

    result["model"] = model
    return result


def report(model: dict):
    print(f"\n\t{GREEN}{model['title']}{RESET}")
    print(f"{model['description']}\n")

    pp(model["stats"])

    if model["plots"]:
        if isinstance(model["model"], DecisionTreeClassifier):
            plt.figure(figsize=(10, 6))
            plot_tree(
                model["model"],
                filled=True,
                feature_names=model["plots"]["feature_names"],
                class_names=list(model["model"].classes_),
            )
            plt.show()
        elif isinstance(
            model["model"], (RandomForestClassifier, GradientBoostingClassifier)
        ):
            # Создаем круговую диаграмму
            plt.pie(
                model["plots"].values(),
                # explode=explode,
                labels=model["plots"].keys(),
                # colors=colors,
                # autopct='%1.1f%%',
                shadow=True,
                # startangle=140,
            )
            plt.title("Важность признаков")
            plt.show()
        else:
            source, args = model["plots"]
            source.plot(**args)
            plt.show()
    if model["Classification report"]:
        display(pd.DataFrame(model["Classification report"]).transpose())
