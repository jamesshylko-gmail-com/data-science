import os
from contextlib import contextmanager
from datetime import date, timedelta

import pandas as pd
import psutil
import seaborn as sns
from colorama import Fore
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib import ticker
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from public.data_loader_utils import load_csv


def load_bitcoin_data() -> pd.DataFrame:
    """
    Загрузка датасета
    """
    return load_csv(r"data\original.csv")


def load_reduced_bitcoin_data() -> pd.DataFrame:
    """
    Загрузка датасета
    """
    return load_csv(r"data\reduced.csv")


def base_report(df: pd.DataFrame, target: str) -> None:
    dict = {
        "rows \u00d7 cols": f"{df.shape[0]}\u00d7{df.shape[1]}",
        "n/a": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum(),
        # "min/max": f"{df.drop(columns=[target]).values.min()}/{df.drop(columns=[target]).values.max()}",
        "0-columns": (df.drop(columns=[target]) == 0).all().sum()
        if target
        else (df == 0).all().sum(),
        "types": f"{', '.join([f'{k}({v})' for k, v in df.dtypes.value_counts().items()])}",
    }
    display(pd.DataFrame([dict]).style.hide(axis="index"))

    str_cols = df.select_dtypes(exclude="number").columns.tolist()
    if str_cols:
        print(f"not numeric: {str_cols}")

    na_series = df.isna().sum().loc[lambda x: x > 0]
    if not na_series.empty:
        print("n/a:")
        display(na_series.to_frame().T.style.hide(axis="index"))


def _get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


@contextmanager
def memory_check():
    base_mb = _get_memory_usage()
    try:
        yield  # This runs the block of code inside the 'with' statement
    finally:
        new_mb = _get_memory_usage()
        print(
            f"{Fore.YELLOW}Memory Increased by: {new_mb - base_mb:.2f} MB{Fore.RESET}"
        )


def outliers(df: pd.DataFrame, title: str):
    Q1 = df[title].quantile(0.25)
    Q3 = df[title].quantile(0.75)

    lower_bound = Q1 - 1.5 * (Q3 - Q1)
    upper_bound = Q3 + 1.5 * (Q3 - Q1)
    return df[(df[title] < lower_bound) | (df[title] > upper_bound)]


def outliers_plot(
    df_sourse: pd.DataFrame, title: str, out_indexes: set[int], top: bool
) -> None:
    def to_date(x, pos):
        return date(2021, 1, 11) + timedelta(days=x)

    df = df_sourse.copy()
    _, ax_main = plt.subplots(figsize=(16, 4), dpi=80)
    df["color_cond"] = df.index.isin(out_indexes)

    ax = sns.scatterplot(
        x=df.index,
        y=title,
        hue="color_cond",
        palette={True: "#E67E22", False: "#2B5C8F"},
        data=df,
        ax=ax_main,
    )
    ymin, ymax = ax.get_ylim()
    test_border = (df.index.max() - df.index.min()) * 0.8
    y_position = ymax - (ymax - ymin) * 0.1 if top else ymin + (ymax - ymin) * 0.1
    ax.axvline(x=test_border, color="gray", linestyle="--", linewidth=1.5)
    ax.text(
        test_border + (-280 if top else 20),  # Координата X чуть правее линии
        y_position,  # Координата Y, где расположится текст (зависит от ваших данных)
        "train/test border",  # Текст подписи
        color="#9B2D30",
        fontsize=14,
        verticalalignment="center",  # Выравнивание по вертикали
    )

    handles, _ = ax_main.get_legend_handles_labels()
    ax_main.legend(title="", handles=handles, labels=["норма", "выбросы"])

    ax.axhline(df[title].mean(), color="blue", linestyle="--", linewidth=2)
    ax_main.xaxis.set_major_formatter(ticker.FuncFormatter(to_date))
    ax_main.set(
        title=f'"{title}" from 2021-01-11 to 2026-01-10', xlabel="", ylabel=title
    )
    ax_main.title.set_fontsize(14)
    ax_main.tick_params(axis="x", rotation=45)
    plt.show()


def feature_plot(df: pd.DataFrame, title: str) -> None:
    def x_k(x, pos):
        if x == 0:
            return "0"
        elif x > 1e6:
            return f"{int(x / 1e6)}M"
        else:
            return f"{int(x / 1000)}к"

    def to_date(x, pos):
        return date(2021, 1, 11) + timedelta(days=x)

    fig = plt.figure(figsize=(16, 4), dpi=80)
    grid = plt.GridSpec(1, 4, width_ratios=[1, 0.05, 2, 0.8])

    ax_left = fig.add_subplot(grid[0, 0])
    ax_main = fig.add_subplot(grid[0, 2])
    ax_right = fig.add_subplot(grid[0, 3], xticklabels=[], yticklabels=[])

    ax = sns.scatterplot(x=df.index, y=title, data=df, ax=ax_main)
    mean_val = df[title].mean()
    ax.axhline(mean_val, color="blue", linestyle="--", linewidth=2)
    sns.histplot(x=df[title], ax=ax_left, kde=True)
    sns.boxplot(y=df[title], ax=ax_right, orient="v")

    ax_left.xaxis.set_major_formatter(ticker.FuncFormatter(x_k))
    ax_main.xaxis.set_major_formatter(ticker.FuncFormatter(to_date))
    ax_left.set(xlabel=title)

    ax_main.set(
        title=f'"{title}" from 2021-01-11 to 2026-01-10', xlabel="", ylabel=title
    )
    ax_main.title.set_fontsize(14)
    ax_main.tick_params(axis="x", rotation=45)
    plt.show()


def feature_importance(X: pd.DataFrame, y: pd.DataFrame) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    df_importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    threshold = 0.95
    df_importance["cumulative"] = (
        df_importance["importance"].cumsum() / df_importance["importance"].sum()
    )
    df_importance["color_group"] = [
        "green" if cum <= threshold else "red" for cum in df_importance["cumulative"]
    ]
    plt.figure(figsize=(12, 4))
    sns.barplot(
        x="importance",
        y="feature",
        data=df_importance,
        hue="color_group",  # Группируем по цвету
        palette={
            c: c for c in df_importance["color_group"].unique()
        },  # Словарь маппинга
        legend=False,  # Скрываем легенду, так как цвета технические
    )
    plt.axvline(
        x=df_importance[df_importance["cumulative"] <= threshold]["importance"].iloc[
            -1
        ],
        color="pink",
        linestyle="--",
    )
    plt.grid(True, axis="x")

    plt.title("Feature Importances")
    plt.xlabel("Relative Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()


def correlogram(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5), dpi=80)
    sns.heatmap(
        df.corr(),
        xticklabels=df.corr().columns,
        yticklabels=df.corr().columns,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".4f",
    )

    plt.title("Correlogram of mtcars", fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


# Testing
if __name__ == "__main__":
    pass
