import pickle
import warnings
from datetime import date, timedelta

import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from colorama import Fore
from IPython.display import display
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from matplotlib import ticker
from optuna import Study
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    r2_score,
)
from sklearn.model_selection import (
    TimeSeriesSplit,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

r"""
- MSE (Mean Squared Error): Средняя квадратичная ошибка. Сильно штрафует за крупные промахи, так как ошибки возводятся в квадрат.
- RMSE (Root Mean Squared Error): Корень из средней квадратичной ошибки. Измеряется в тех же единицах, что и исходные данные, что удобно для интерпретации.
- MAE (Mean Absolute Error): Средняя абсолютная ошибка. Метрика устойчива к выбросам, так как отклонения берутся по модулю без возведения в степень.
- MAPE (Mean Absolute Percentage Error): Средняя абсолютная ошибка в процентах. Показывает ошибку относительно реального значения, что удобно для сравнения разных наборов данных.
- \(R^{2}\) (Coefficient of Determination): Коэффициент детерминации. Показывает долю дисперсии, объясненную моделью (идеально равен 1, но может уходить в минус для плохих моделей).

"""

global_shuffle = True


class Notes:
    def __init__(
        self,
        losses=None,
        r2=None,
        mse=None,
        rmse=None,
        mae=None,
        mape=None,
        r2_std=None,
        mse_std=None,
        rmse_std=None,
        mae_std=None,
        mape_std=None,
    ):
        self.losses = losses
        self.r2 = r2
        self.mse = mse
        self.rmse = rmse
        self.mae = mae
        self.mape = mape
        self.r2_std = r2_std
        self.mse_std = mse_std
        self.rmse_std = rmse_std
        self.mae_std = mae_std
        self.mape_std = mape_std


class RegressionResult:
    def __init__(
        self,
        title: str,
        model,
        train_notes: Notes,
        val_notes: Notes,
        test_notes: Notes = None,
        is_tuned: bool = False,
    ):
        self.title = title
        self.model = model
        self.train_notes = train_notes
        self.val_notes = val_notes
        self.test_notes = test_notes
        self.is_tuned = is_tuned


# @log_time
def regression(
    model, df: pd.DataFrame, target_column: str, title: str, opts: dict
) -> RegressionResult:
    warnings.filterwarnings("ignore")
    if opts:
        model.set_params(**opts)
    X, y = df.drop(columns=[target_column]), df[target_column]
    scoring = {
        "R2": "r2",
        "MSE": "neg_mean_squared_error",
        "RMSE": "neg_root_mean_squared_error",
        "MAE": "neg_mean_absolute_error",
        "MAPE": "neg_mean_absolute_percentage_error",
    }

    if isinstance(model, LinearRegression):
        return lin_reg_calc(X, y, title, scoring)
    elif isinstance(model, DecisionTreeRegressor):
        return tree_calc(X, y, title, scoring, create_dec_tree)
    elif isinstance(model, RandomForestRegressor):
        return tree_calc(X, y, title, scoring, create_ran_for)
    elif isinstance(model, Ridge):
        return ridge_reg_calc(X, y, title, scoring)
    elif isinstance(model, XGBRegressor):
        return boost_calc(X, y, title, scoring, create_xgb)
    elif isinstance(model, GradientBoostingRegressor):
        return boost_calc(X, y, title, scoring, create_gboost)
    elif isinstance(model, LGBMRegressor):
        return boost_calc(X, y, title, scoring, create_lgbm)
    elif isinstance(model, ExtraTreesRegressor):
        return boost_calc(X, y, title, scoring, create_xgb_sl)
    elif isinstance(model, AdaBoostRegressor):
        return boost_calc(X, y, title, scoring, create_ada_boost)
    elif isinstance(model, CatBoostRegressor):
        return boost_calc(X, y, title, scoring, create_cat_boost)
    else:
        return None


def lin_reg_calc(X, y, title, scoring) -> RegressionResult:
    print(f"{Fore.GREEN}Обучение LinearRegression...{Fore.RESET}")
    if global_shuffle:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

    def create_poly_model(degree):
        return Pipeline(
            [
                ("poly", PolynomialFeatures(degree=degree)),
                ("linear", LinearRegression()),
            ]
        )

    def train_poly_with_cv(degree, X_train, y_train):
        model = create_poly_model(degree)

        # Кросс-валидация
        cv_scores = cross_validate(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=TimeSeriesSplit(n_splits=5),
            scoring=scoring,
            return_train_score=True,
            return_estimator=True,  # Чтобы получить обученные модели
            n_jobs=-1,
        )

        # Обучаем финальную модель на всех данных
        final_model = create_poly_model(degree)
        final_model.fit(X_train, y_train)

        return {
            "cv_scores": cv_scores,
            "final_model": final_model,
            "train_r2_mean": np.mean(cv_scores["train_R2"]),
            "train_r2_std": np.std(cv_scores["train_R2"]),
            "val_r2_mean": np.mean(cv_scores["test_R2"]),
            "val_r2_std": np.std(cv_scores["test_R2"]),
            "gap_mean": np.mean(cv_scores["train_R2"] - cv_scores["test_R2"]),
        }

    # Тестируем разные степени
    degrees = [1, 2, 3, 5, 7, 10]
    _, axes = plt.subplots(2, 3, figsize=(15, 10))

    cv_results = []

    for idx, degree in enumerate(degrees):
        ax = axes[idx // 3, idx % 3]

        result = train_poly_with_cv(degree, X_train, y_train)
        cv_results.append(result)

        X_plot = X.copy()
        y_plot = result["final_model"].predict(X_plot)
        test_score = r2_score(y_test, result["final_model"].predict(X_test))

        def to_date(x, pos):
            return date(2021, 1, 11) + timedelta(days=(x - 1610323) // 86400)

        # Визуализация
        ax.scatter(X_train["date"], y_train, label="Train", s=5, alpha=0.7)
        ax.scatter(X_test["date"], y_test, label="Test", s=5, alpha=0.7, color="green")
        ax.plot(X_plot["date"], y_plot, "r-", linewidth=2, label=f"Degree {degree}")

        # Отображаем результаты кросс-валидации
        ax.set_title(
            f"Степень {degree}\n"
            f"CV Train R²={result['train_r2_mean']:.3f}±{result['train_r2_std']:.3f}\n"
            f"CV Val R²={result['val_r2_mean']:.3f}±{result['val_r2_std']:.3f}\n"
            f"Test R²={test_score:.3f}"
        )
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(to_date))
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        _, _, color = diagnose_model(
            result["train_r2_mean"],
            result["train_r2_std"],
            result["val_r2_mean"],
            result["val_r2_std"],
        )
        # Цветовая индикация переобучения по CV
        ax.set_facecolor(color)

    plt.suptitle("Влияние степени полинома на переобучение", fontsize=14)
    plt.tight_layout()
    plt.show()

    best_degree = degrees[np.argmin([r["gap_mean"] for r in cv_results])]
    best_result = cv_results[degrees.index(best_degree)]
    cv_scores = best_result["cv_scores"]

    # Вывод таблицы результатов
    print("\n" + "=" * 80)
    print(f"РЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ (CV=5). Степень: {best_degree}")
    print("=" * 80)
    print(
        f"{'Степень':<8} {'Train R² (CV)':<15} {'Val R² (CV)':<15} {'Разрыв':<12} {'Test R²':<12} {'Вердикт':<15}"
    )
    print("-" * 80)

    for degree, result in zip(degrees, cv_results):
        test_score = r2_score(y_test, result["final_model"].predict(X_test))
        gap = result["gap_mean"]
        verdict, _, _ = diagnose_model(
            result["train_r2_mean"],
            result["train_r2_std"],
            result["val_r2_mean"],
            result["val_r2_std"],
        )

        print(
            f"{degree:<8} "
            f"{result['train_r2_mean']:.3f} ± {result['train_r2_std']:.3f}    "
            f"{result['val_r2_mean']:.3f} ± {result['val_r2_std']:.3f}    "
            f"{gap:.3f}       "
            f"{test_score:.3f}    "
            f"{verdict}"
        )

    print("\n" + "=" * 80)
    print("ИТОГО:")

    # Находим оптимальную степень
    optimal_degree = degrees[np.argmin([abs(r["gap_mean"] - 0.05) for r in cv_results])]
    print(f"✅ Оптимальная степень (минимальный разрыв): {optimal_degree}")

    # Вывод лучшей модели
    best_model = cv_results[degrees.index(best_degree)]["final_model"]
    print(f"\n📊 Лучшая модель: степень {best_degree}")
    print(
        f"   - Средний R² на валидации: {best_result['val_r2_mean']:.3f} ± {best_result['val_r2_std']:.3f}"
    )
    print(
        f"   - R² на тестовой выборке: {r2_score(y_test, best_model.predict(X_test)):.3f}"
    )
    reg_res = RegressionResult(
        title=title,
        model=best_model,
        train_notes=create_notes(cv_scores, "train"),
        val_notes=create_notes(cv_scores, "test"),
    )
    return reg_res


def create_ran_for(max_depth):
    return RandomForestRegressor(
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,  # 'sqrt', 'log2', или число
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,  # можно включить для оценки OOB
        n_jobs=-1,
        random_state=42,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
    )


def create_dec_tree(max_depth):
    return DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,  # None = все признаки
        random_state=42,
        max_leaf_nodes=None,  # None = без ограничений
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,  # параметр сложности для обрезки (cost-complexity pruning)
        splitter="best",  # 'best' или 'random'
    )


def tree_calc(X, y, title, scoring, create_model) -> RegressionResult:
    print(f"{Fore.GREEN}Обучение {title}...{Fore.RESET}")

    if global_shuffle:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

    def evaluate_with_cv(max_depth, X, y, scoring):
        model = create_model(max_depth)

        cv_scores = cross_validate(
            estimator=model,
            X=X,
            y=y,
            cv=TimeSeriesSplit(n_splits=5),
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True,
            return_estimator=True,
        )

        return {
            "cv_scores": cv_scores,
            "model": model,
        }

    cv_results = []

    print("\n" + "-" * 70)
    print("РЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ ПО max_depth")
    print("-" * 70)
    print(
        f"{'Depth':<10} {'Train R²':<12} {'Val R²':<12} {'Train MSE':<12} {'Val MSE':<12} {'Разрыв R²':<10} {'Статус':<12}"
    )
    print("-" * 70)

    depth_range = range(1, 21)

    for max_depth in depth_range:
        results = evaluate_with_cv(max_depth, X_train, y_train, scoring)
        r2, t_r2, mse, t_mse = (
            results["cv_scores"]["train_R2"],
            results["cv_scores"]["test_R2"],
            results["cv_scores"]["train_MSE"],
            results["cv_scores"]["test_MSE"],
        )

        verdict, comment, _ = diagnose_model(
            np.mean(r2), np.mean(t_r2), np.std(r2), np.std(t_r2)
        )

        print(
            f"{max_depth:<10} "
            f"{np.mean(r2):.3f} ± {np.std(r2):.3f} "
            f"{np.mean(t_r2):.3f} ± {np.std(t_r2):.3f} "
            f"{-np.mean(mse):.3f} "
            f"{-np.mean(t_mse):.3f} "
            f"{np.mean(r2) - np.mean(t_r2):.3f}    "
            f"{verdict}"
        )

        cv_results.append(
            {
                "max_depth": max_depth,
                "train_r2_mean": np.mean(r2),
                "train_r2_std": np.std(r2),
                "test_r2_mean": np.mean(t_r2),
                "test_r2_std": np.std(t_r2),
                "train_mse_mean": np.mean(mse),
                "test_mse_mean": np.mean(t_mse),
                "gap": np.mean(r2) - np.mean(t_r2),
                "status": f"{verdict} {comment}",
                "results": results,
            }
        )

    _, axes = plt.subplots(1, 2, figsize=(15, 5))

    train_r2_means = [r["train_r2_mean"] for r in cv_results]
    train_r2_stds = [r["train_r2_std"] for r in cv_results]
    test_r2_means = [r["test_r2_mean"] for r in cv_results]
    test_r2_stds = [r["test_r2_std"] for r in cv_results]

    ax1 = axes[0]
    ax1.plot(
        depth_range,
        train_r2_means,
        "b-o",
        label="Train R²",
        linewidth=2,
        markersize=8,
    )
    ax1.fill_between(
        depth_range,
        np.array(train_r2_means) - np.array(train_r2_stds),
        np.array(train_r2_means) + np.array(train_r2_stds),
        color="blue",
        alpha=0.2,
    )
    ax1.plot(
        depth_range,
        test_r2_means,
        "r-o",
        label="Validation R²",
        linewidth=2,
        markersize=8,
    )
    ax1.fill_between(
        depth_range,
        np.array(test_r2_means) - np.array(test_r2_stds),
        np.array(test_r2_means) + np.array(test_r2_stds),
        color="red",
        alpha=0.2,
    )

    # Отмечаем лучшую глубину
    best_idx = np.argmax(test_r2_means)
    best_depth = cv_results[best_idx]["max_depth"]

    # Обучаем финальную модель с лучшей глубиной
    final_result = evaluate_with_cv(best_depth, X_train, y_train, scoring)
    final_result["model"].fit(X_train, y_train)
    y_train_pred = final_result["model"].predict(X_train)
    y_test_pred = final_result["model"].predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    ax1.axvline(
        x=best_depth,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Best max_depth = {best_depth}",
    )
    ax1.axhline(
        y=0.5, color="orange", linestyle="--", alpha=0.5, label="Порог хорошей модели"
    )

    ax1.set_xlabel("max_depth", fontsize=12)
    ax1.set_ylabel("R² Score", fontsize=12)
    ax1.set_title("Кривая валидации (R²)", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Добавляем аннотацию с оптимальным значением
    ax1.annotate(
        f"Max R² = {train_r2:.4f}",
        xy=(best_depth, train_r2),
        xytext=(best_depth + 1, train_r2 - 0.05),
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 1.5},
        fontsize=10,
        color="red",
    )

    ax2 = axes[1]
    ax2.scatter(
        y_train,
        y_train_pred,
        alpha=0.6,
        color="blue",
        label=f"Train (R²={train_r2:.3f})",
    )
    ax2.scatter(
        y_test, y_test_pred, alpha=0.6, color="orange", label=f"Test (R²={test_r2:.3f})"
    )
    ax2.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
    ax2.set_xlabel("Истинные значения", fontsize=12)
    ax2.set_ylabel("Предсказанные значения", fontsize=12)
    ax2.set_title("Предсказание vs Факт", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    cv_scores = final_result["cv_scores"]
    reg_res = RegressionResult(
        title=title,
        model=final_result["model"],
        train_notes=create_notes(cv_scores, "train"),
        val_notes=create_notes(cv_scores, "test"),
    )
    return reg_res


def ridge_reg_calc(X, y, title, scoring) -> RegressionResult:
    print(f"{Fore.GREEN}Обучение Ridge...{Fore.RESET}")
    if global_shuffle:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

    def evaluate_ridge_with_cv(alpha, X, y, scoring, scale=True):
        model = (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha, random_state=42)),
                ]
            )
            if scale
            else Ridge(alpha=alpha, random_state=42)
        )

        cv_scores = cross_validate(
            estimator=model,
            X=X,
            y=y,
            cv=TimeSeriesSplit(n_splits=5),
            scoring=scoring,
            return_train_score=True,
            return_estimator=True,
            n_jobs=-1,
        )

        return {
            "train_r2": cv_scores["train_R2"],
            "test_r2": cv_scores["test_R2"],
            "train_mse": -cv_scores["train_MSE"],
            "test_mse": -cv_scores["test_MSE"],
            "estimators": cv_scores["estimator"],
            "cv_scores": cv_scores,
            "model": model,
        }

    alphas = [
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        0.1,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,
    ]
    cv_results = []

    print("\n" + "-" * 70)
    print("РЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ ПО ALPHA")
    print("-" * 70)
    print(
        f"{'Alpha':<10} {'Train R²':<12} {'Val R²':<12} {'Train MSE':<12} {'Val MSE':<12} {'Разрыв R²':<10} {'Статус':<12}"
    )
    print("-" * 70)

    for alpha in alphas:
        results = evaluate_ridge_with_cv(alpha, X_train, y_train, scoring)
        r2, t_r2, mse, t_mse = (
            results["train_r2"],
            results["test_r2"],
            results["train_mse"],
            results["test_mse"],
        )
        verdict, comment, _ = diagnose_model(
            np.mean(r2), np.mean(t_r2), np.std(r2), np.std(t_r2)
        )
        print(
            f"{alpha:<10.1e} "
            f"{np.mean(r2):.3f} ± {np.std(r2):.3f} "
            f"{np.mean(t_r2):.3f} ± {np.std(t_r2):.3f} "
            f"{np.mean(mse):.3f} "
            f"{np.mean(t_mse):.3f} "
            f"{np.mean(r2) - np.mean(t_r2):.3f}    "
            f"{verdict}"
        )

        cv_results.append(
            {
                "alpha": alpha,
                "train_r2_mean": np.mean(r2),
                "train_r2_std": np.std(r2),
                "test_r2_mean": np.mean(t_r2),
                "test_r2_std": np.std(t_r2),
                "train_mse_mean": np.mean(mse),
                "test_mse_mean": np.mean(t_mse),
                "gap": np.mean(r2) - np.mean(t_r2),
                "status": f"{verdict} {comment}",
                "results": results,
            }
        )

    _, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax1 = axes[0]
    alphas_log = np.log10(alphas)

    train_r2_means = [r["train_r2_mean"] for r in cv_results]
    train_r2_stds = [r["train_r2_std"] for r in cv_results]
    test_r2_means = [r["test_r2_mean"] for r in cv_results]
    test_r2_stds = [r["test_r2_std"] for r in cv_results]

    ax1.plot(
        alphas_log, train_r2_means, "b-o", label="Train R²", linewidth=2, markersize=8
    )
    ax1.fill_between(
        alphas_log,
        np.array(train_r2_means) - np.array(train_r2_stds),
        np.array(train_r2_means) + np.array(train_r2_stds),
        color="blue",
        alpha=0.2,
    )
    ax1.plot(
        alphas_log,
        test_r2_means,
        "r-o",
        label="Validation R²",
        linewidth=2,
        markersize=8,
    )
    ax1.fill_between(
        alphas_log,
        np.array(test_r2_means) - np.array(test_r2_stds),
        np.array(test_r2_means) + np.array(test_r2_stds),
        color="red",
        alpha=0.2,
    )
    # Отмечаем лучший alpha
    best_idx = np.argmax(test_r2_means)
    best_alpha = cv_results[best_idx]["alpha"]
    final_result = evaluate_ridge_with_cv(best_alpha, X, y, scoring)

    final_result["model"].fit(X_train, y_train)
    y_train_pred = final_result["model"].predict(X_train)
    y_test_pred = final_result["model"].predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)

    ax1.axvline(
        x=np.log10(best_alpha),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Best alpha = {best_alpha:.1e}",
    )
    ax1.axhline(
        y=0.5, color="orange", linestyle="--", alpha=0.5, label="Порог хорошей модели"
    )

    ax1.set_xlabel("log10(Alpha)")
    ax1.set_ylabel("R² Score")
    ax1.set_title("Кривая валидации по R²")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(y_train, y_train_pred, alpha=0.6, color="blue", label="Train")
    ax2.scatter(y_test, y_test_pred, alpha=0.6, color="green", label="Test")
    ax2.plot(
        [y_train.min(), y_train.max()],
        [y_train.min(), y_train.max()],
        "r--",
        linewidth=2,
        label="Perfect prediction",
    )
    ax2.set_xlabel("Истинные значения")
    ax2.set_ylabel("Предсказанные значения")
    ax2.set_title(f"Train: R²={train_r2:.3f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("СТАТИСТИКА ПО ALPHA")
    print("=" * 70)

    # Находим оптимальный alpha по разным критериям
    best_r2_alpha = alphas[np.argmax(test_r2_means)]
    best_gap_alpha = alphas[np.argmin([abs(r["gap"]) for r in cv_results])]

    print(
        f"\nОптимальный alpha по R²: {best_r2_alpha:.1e} (R²={max(test_r2_means):.4f})"
    )
    print(
        f"Оптимальный alpha по разрыву: {best_gap_alpha:.1e} (Gap={min([abs(r['gap']) for r in cv_results]):.4f})"
    )

    cv_scores = final_result["cv_scores"]
    reg_res = RegressionResult(
        title=title,
        model=final_result["model"],
        train_notes=create_notes(cv_scores, "train"),
        val_notes=create_notes(cv_scores, "test"),
    )
    return reg_res


def create_xgb(n_estimators):
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


def create_xgb_sl(n_estimators):
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=6,  # можно настроить
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,  # или 'log2', число
        bootstrap=True,  # использование bootstrap для уменьшения переобучения
        oob_score=False,  # можно включить для оценки OOB
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )


def create_cat_boost(n_estimators):
    return CatBoostRegressor(
        iterations=n_estimators,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3.0,
        border_count=254,
        verbose=False,
        random_seed=42,
        loss_function="RMSE",  # Рекомендуется оставить RMSE для обучения
        eval_metric="R2",  # Используем R2 для оценки и ранней остановки
        od_type="Iter",
        od_wait=20,
        allow_writing_files=False,
        thread_count=-1,
    )


def create_gboost(n_estimators):
    return GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        min_samples_split=2,
        min_samples_leaf=1,
    )


def create_lgbm(n_estimators):
    return LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )


def create_ada_boost(n_estimators):
    return AdaBoostRegressor(
        n_estimators=n_estimators,
        learning_rate=1.0,  # можно настроить: 0.5, 0.1, 1.0
        loss="linear",  # 'linear', 'square', 'exponential'
        random_state=42,
    )


def boost_calc(X, y, title, scoring, create_model) -> RegressionResult:
    print(f"{Fore.GREEN}Обучение {title}...{Fore.RESET}")

    if global_shuffle:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

    def evaluate_boost_with_cv(n_estimators, X, y, scoring):
        model = create_model(n_estimators)

        cv_scores = cross_validate(
            estimator=model,
            X=X,
            y=y,
            cv=TimeSeriesSplit(n_splits=5),
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True,
            return_estimator=True,
        )

        return {
            "cv_scores": cv_scores,
            "model": model,
        }

    cv_results = []

    print("\n" + "-" * 70)
    print("РЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ ПО n_estimators")
    print("-" * 70)
    print(
        f"{'Trees':<10} {'Train R²':<12} {'Val R²':<12} {'Train MSE':<12} {'Val MSE':<12} {'Разрыв R²':<10} {'Статус':<12}"
    )
    print("-" * 70)

    estimation_range = np.arange(10, 210, 10)
    for n_estimators in estimation_range:
        results = evaluate_boost_with_cv(n_estimators, X_train, y_train, scoring)
        r2, t_r2, mse, t_mse = (
            results["cv_scores"]["train_R2"],
            results["cv_scores"]["test_R2"],
            results["cv_scores"]["train_MSE"],
            results["cv_scores"]["test_MSE"],
        )
        verdict, comment, _ = diagnose_model(
            np.mean(r2), np.mean(t_r2), np.std(r2), np.std(t_r2)
        )
        print(
            f"{n_estimators:<10} "
            f"{np.mean(r2):.3f} ± {np.std(r2):.3f} "
            f"{np.mean(t_r2):.3f} ± {np.std(t_r2):.3f} "
            f"{np.mean(mse):.3f} "
            f"{np.mean(t_mse):.3f} "
            f"{np.mean(r2) - np.mean(t_r2):.3f}    "
            f"{verdict}"
        )

        cv_results.append(
            {
                "n_estimators": n_estimators,
                "train_r2_mean": np.mean(r2),
                "train_r2_std": np.std(r2),
                "test_r2_mean": np.mean(t_r2),
                "test_r2_std": np.std(t_r2),
                "train_mse_mean": np.mean(mse),
                "test_mse_mean": np.mean(t_mse),
                "gap": np.mean(r2) - np.mean(t_r2),
                "status": f"{verdict} {comment}",
                "results": results,
            }
        )

    _, axes = plt.subplots(1, 2, figsize=(15, 5))

    train_r2_means = [r["train_r2_mean"] for r in cv_results]
    train_r2_stds = [r["train_r2_std"] for r in cv_results]
    test_r2_means = [r["test_r2_mean"] for r in cv_results]
    test_r2_stds = [r["test_r2_std"] for r in cv_results]

    ax1 = axes[0]
    ax1.plot(
        estimation_range,
        train_r2_means,
        "b-o",
        label="Train R²",
        linewidth=2,
        markersize=8,
    )
    ax1.fill_between(
        estimation_range,
        np.array(train_r2_means) - np.array(train_r2_stds),
        np.array(train_r2_means) + np.array(train_r2_stds),
        color="blue",
        alpha=0.2,
    )
    ax1.plot(
        estimation_range,
        test_r2_means,
        "r-o",
        label="Validation R²",
        linewidth=2,
        markersize=8,
    )
    ax1.fill_between(
        estimation_range,
        np.array(test_r2_means) - np.array(test_r2_stds),
        np.array(test_r2_means) + np.array(test_r2_stds),
        color="red",
        alpha=0.2,
    )

    # Отмечаем лучший estimation
    best_idx = np.argmax(test_r2_means)
    best_estimation = cv_results[best_idx]["n_estimators"]

    final_result = evaluate_boost_with_cv(best_estimation, X_train, y_train, scoring)
    final_result["model"].fit(X_train, y_train)
    y_train_pred = final_result["model"].predict(X_train)
    y_test_pred = final_result["model"].predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)

    ax1.axvline(
        x=best_estimation,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Best estimation = {best_estimation:.1e}",
    )
    ax1.axhline(
        y=0.5, color="orange", linestyle="--", alpha=0.5, label="Порог хорошей модели"
    )

    ax1.set_xlabel("Number of Trees (n_estimators)", fontsize=12)
    ax1.set_ylabel("R² Score", fontsize=12)
    ax1.set_title("Кривая валидации (R²)", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Добавляем аннотацию с оптимальным значением
    ax1.annotate(
        f"Max R² = {train_r2:.4f}",
        xy=(best_estimation, train_r2),
        xytext=(best_estimation + 20, train_r2 - 0.05),
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 1.5},
        fontsize=10,
        color="red",
    )

    ax2 = axes[1]
    ax2.scatter(y_train, y_train_pred, alpha=0.6, color="blue", label="Train")
    ax2.scatter(y_test, y_test_pred, alpha=0.6, color="orange", label="Test")
    ax2.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
    ax2.set_xlabel("Истинные значения", fontsize=12)
    ax2.set_ylabel("Предсказанные значения", fontsize=12)
    ax2.set_title(f"Train: R²={train_r2:.3f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    cv_scores = final_result["cv_scores"]
    reg_res = RegressionResult(
        title=title,
        model=final_result["model"],
        train_notes=create_notes(cv_scores, "train"),
        val_notes=create_notes(cv_scores, "test"),
    )
    return reg_res


def create_notes(cv_scores, prefix: str) -> Notes:
    return Notes(
        losses=cv_scores[f"{prefix}_R2"],
        r2=cv_scores[f"{prefix}_R2"].mean(),
        mse=-cv_scores[f"{prefix}_MSE"].mean(),
        rmse=-cv_scores[f"{prefix}_RMSE"].mean(),
        mae=-cv_scores[f"{prefix}_MAE"].mean(),
        mape=-cv_scores[f"{prefix}_MAPE"].mean(),
        r2_std=cv_scores[f"{prefix}_R2"].std(),
        mse_std=cv_scores[f"{prefix}_MSE"].std(),
        rmse_std=cv_scores[f"{prefix}_RMSE"].std(),
        mae_std=cv_scores[f"{prefix}_MAE"].std(),
        mape_std=cv_scores[f"{prefix}_MAPE"].std(),
    )


def diagnose_model(train_r2_mean, val_r2_mean, train_r2_std=None, val_r2_std=None):
    """
    Диагностика состояния модели на основе метрик R²

    Parameters:
    - train_r2_mean: средний R² на обучении (CV)
    - val_r2_mean: средний R² на валидации (CV)
    - train_r2_std: стандартное отклонение R² на обучении
    - val_r2_std: стандартное отклонение R² на валидации
    """
    gap = train_r2_mean - val_r2_mean
    val_r2 = val_r2_mean
    # Цветовая индикация переобучения по CV
    color_overfit = "#FFE5E5"  # Красноватый - переобучение
    color_light_overfit = "#FFF5E5"  # Желтоватый - легкое переобучение
    color_underfit = "#C7E3EC"  # Голубоватый - недообучение
    color_norm = "#E5FFE5"  # Зеленоватый - хорошее обобщение
    color_unstable = "#F4C2E1"  # Розоватый - нестабильная можель

    # 1. Проверка на недообучение
    if val_r2 < 0.3:
        return (
            "🌀 НЕДООБУЧЕНИЕ",
            "Модель слишком простая, R² на валидации < 0.3",
            color_underfit,
        )
    elif val_r2 < 0.5:
        return (
            "⚠️ Слабое обучение",
            "R² на валидации между 0.3 и 0.5, стоит улучшить модель",
            color_underfit,
        )

    # 2. Проверка на переобучение
    if gap > 0.15:
        return (
            "❌ ПЕРЕОБУЧЕНИЕ",
            f"Большой разрыв {gap:.3f} между train и val",
            color_overfit,
        )
    elif gap > 0.08:
        return (
            "⚠️ Легкое переобучение",
            f"Разрыв {gap:.3f} между train и val",
            color_light_overfit,
        )

    # 3. Проверка стабильности (если есть std)
    if val_r2_std is not None and val_r2_std > 0.1:
        return (
            "⚡ Нестабильная модель",
            f"Высокая вариация R²: ±{val_r2_std:.3f}",
            color_unstable,
        )

    # 4. Отличная модель
    if val_r2 > 0.8 and gap < 0.05:
        return "✅ ОТЛИЧНО", "Высокое качество и хорошее обобщение", color_norm
    elif gap < 0.05:
        return "✅ Хорошо", "Модель сбалансирована", color_norm
    else:
        return "✅ Нормально", "Приемлемое качество", color_norm


def model_training_logging(reg_res: RegressionResult) -> None:
    _, ax = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={"width_ratios": [1, 2]})

    r2_str = (
        str(round(reg_res.val_notes.r2, 4))
        + " ± "
        + str(round(reg_res.val_notes.r2_std, 4))
    )
    mse_str = (
        str(round(reg_res.val_notes.mse, 4))
        + " ± "
        + str(round(reg_res.val_notes.mse_std, 4))
    )
    ax[0].axis("off")
    ax[0].text(0.05, 0.85, f"{reg_res.title:^50}", color="green")
    verdict, comment, _ = diagnose_model(
        reg_res.train_notes.r2,
        reg_res.train_notes.r2_str,
        reg_res.val_notes.r2,
        reg_res.val_notes.r2_str,
    )
    ax[0].text(
        0.05,
        0.45,
        f"train | val :\n \
        r2 : {reg_res.train_notes.r2:.4f} | {r2_str}\n \
        mse : {reg_res.train_notes.mse:.4f} | {mse_str}\n \
        state : {verdict} {comment}",
    )

    ax[1].plot(reg_res.train_notes.r2, label="Train Loss", color="blue")
    ax[1].plot(reg_res.val_notes.r2, label="Val Loss", color="orange")

    ax[1].set_title("График оценки переобучения")
    ax[1].set_xlabel("Эпоха")
    ax[1].set_ylabel("Loss (Потери)")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


def _print_param(model) -> None:
    print("\nВсе параметры модели:")
    for key, value in model.get_params().items():
        print(f"{key}: {value}")


def summary_report(models: list[RegressionResult]) -> None:
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.float_format", "{:.4f}".format)

    models.sort(key=lambda x: x.train_notes.r2, reverse=True)
    result = {}
    for model in models:
        verdict, comment, _ = diagnose_model(
            train_r2_mean=model.train_notes.r2,
            train_r2_std=model.train_notes.r2_std,
            val_r2_mean=model.val_notes.r2,
            val_r2_std=model.val_notes.r2_std,
        )
        result[model.title] = {
            "R² train": f"{model.train_notes.r2:.3f}",
            "R² val": f"{model.val_notes.r2:.3f} ± {model.val_notes.r2_std:.3f}",
            "MSE train": f"{model.train_notes.mse:.5f}",
            "MSE val": f"{model.val_notes.mse:.5f} ± {model.val_notes.mae_std:.5f}",
            "state": f"{verdict} {comment}",
        }
    df = pd.DataFrame.from_dict(result, orient="index")

    # Функция для подсветки первых 5 строк
    def highlight_first_five(row):
        if row.name in df.index[:5]:
            return ["color: lightgreen"] * len(row)
        return [""] * len(row)

    display(df.style.apply(highlight_first_five, axis=1))


def super_report(
    models: list[RegressionResult], best_models: list[RegressionResult]
) -> str | None:
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.float_format", "{:.4f}".format)
    summary = []
    # for item in models:
    #     summary.append(ExtendedRegressionResult.from_parent(item, False))
    # for item in best_models:
    #     summary.append(ExtendedRegressionResult.from_parent(item, True))

    summary.sort(key=lambda x: x.r2, reverse=True)
    result = {}
    for model in summary:
        result[f"{model.title}_tuned" if model.is_best else model.title] = {
            "R²": model.r2,
            "MSE": model.mse,
            "RMSE": model.rmse,
            "MAE": model.mae,
            "MAPE": model.mape,
            "Best": model.is_best,
        }
    df = pd.DataFrame.from_dict(result, orient="index")

    def color_rows(row):
        color = "color: #F08080" if row["Best"] else "color: #7BC8F6"
        return [color] * len(row)

    display(df.style.apply(color_rows, axis=1))
    return f"{summary[0].title}_tuned" if summary[0].is_best else summary[0].title


def optuna_report(studies: list[dict]) -> None:
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.float_format", "{:.4f}".format)

    summary = {}
    for item in studies:
        summary[item["title"]] = {
            "R²": item["study"].best_value,
            "Best params": item["study"].best_params,
        }
    df = pd.DataFrame.from_dict(summary, orient="index")
    display(df)


def report(df: pd.DataFrame, title: str, model: RegressionResult) -> None:
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.float_format", "{:.4f}".format)

    x_test, y_test = df.drop(columns=[title]), df[title]
    y_pred = model.model.predict(x_test)

    print(f"\n\t{Fore.GREEN}{model.title}{Fore.RESET}")
    display(
        pd.DataFrame(
            [
                {
                    "R²": model.r2,
                    "R² (test)": r2_score(y_test, y_pred),
                    "MSE": model.mse,
                    "RMSE": model.rmse,
                    "MAE": model.mae,
                    "MAPE": model.mape,
                }
            ]
        ).style.hide(axis="index")
    )

    plt.figure(figsize=(12, 4))
    sns.scatterplot(x=y_test, y=y_pred, color="purple", alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Фактические значения")
    plt.ylabel("Предсказанные значения")
    plt.title(f"Фактические vs Предсказанные значения ({model.title})")
    plt.show()


def optuna_plots(study: Study) -> None:
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_param_importances(study).show()
    # optuna.visualization.plot_slice(study).show()
    optuna.visualization.plot_contour(study).show()
    optuna.visualization.plot_parallel_coordinate(study).show()


def pick(data, name: str) -> None:
    with open(f"./data/pick/{name}.pkl", "wb") as f:
        pickle.dump(data, f)
