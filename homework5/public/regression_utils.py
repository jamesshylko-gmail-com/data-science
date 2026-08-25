import pickle
import warnings

import optuna
import pandas as pd
import seaborn as sns
from colorama import Fore
from IPython.display import display
from matplotlib import pyplot as plt
from optuna import Study
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    KFold,
    cross_val_predict,
)

r"""
- MSE (Mean Squared Error): Средняя квадратичная ошибка. Сильно штрафует за крупные промахи, так как ошибки возводятся в квадрат.
- RMSE (Root Mean Squared Error): Корень из средней квадратичной ошибки. Измеряется в тех же единицах, что и исходные данные, что удобно для интерпретации.
- MAE (Mean Absolute Error): Средняя абсолютная ошибка. Метрика устойчива к выбросам, так как отклонения берутся по модулю без возведения в степень.
- MAPE (Mean Absolute Percentage Error): Средняя абсолютная ошибка в процентах. Показывает ошибку относительно реального значения, что удобно для сравнения разных наборов данных.
- \(R^{2}\) (Coefficient of Determination): Коэффициент детерминации. Показывает долю дисперсии, объясненную моделью (идеально равен 1, но может уходить в минус для плохих моделей).

"""


class RegressionResult:
    def __init__(self, title, model, r2, mse, rmse, mae, mape):
        self.title = title
        self.model = model
        self.r2 = r2
        self.mse = mse
        self.rmse = rmse
        self.mae = mae
        self.mape = mape


class ExtendedRegressionResult(RegressionResult):
    def __init__(self, title, model, r2, mse, rmse, mae, mape, is_best=False):
        super().__init__(title, model, r2, mse, rmse, mae, mape)
        self.is_best = is_best

    @classmethod
    def from_parent(cls, parent_obj, is_best=False):
        """Создает объект ExtendedRegressionResult из объекта RegressionResult."""
        return cls(
            title=parent_obj.title,
            model=parent_obj.model,
            r2=parent_obj.r2,
            mse=parent_obj.mse,
            rmse=parent_obj.rmse,
            mae=parent_obj.mae,
            mape=parent_obj.mape,
            is_best=is_best,
        )


# @log_time
def regression(
    model, df: pd.DataFrame, target_column: str, title: str, opts: dict
) -> RegressionResult:
    warnings.filterwarnings("ignore")
    if opts:
        model.set_params(**opts)
    X, y = df.drop(columns=[target_column]), df[target_column]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=kf)
    model.fit(X, y)

    return RegressionResult(
        title=title,
        model=model,
        r2=r2_score(y, y_pred),
        mse=mean_squared_error(y, y_pred),
        rmse=root_mean_squared_error(y, y_pred),
        mae=mean_absolute_error(y, y_pred),
        mape=mean_absolute_percentage_error(y, y_pred),
    )


def _print_param(model) -> None:
    print("\nВсе параметры модели:")
    for key, value in model.get_params().items():
        print(f"{key}: {value}")


def summary_report(models: list[RegressionResult]) -> None:
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.float_format", "{:.4f}".format)

    models.sort(key=lambda x: x.r2, reverse=True)
    result = {}
    for model in models:
        result[model.title] = {
            "R²": model.r2,
            "MSE": model.mse,
            "RMSE": model.rmse,
            "MAE": model.mae,
            "MAPE": model.mape,
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
    for item in models:
        summary.append(ExtendedRegressionResult.from_parent(item, False))
    for item in best_models:
        summary.append(ExtendedRegressionResult.from_parent(item, True))

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
