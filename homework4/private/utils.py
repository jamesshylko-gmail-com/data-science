import os
from contextlib import contextmanager

import pandas as pd
import psutil
from IPython.display import display
from keras.datasets import mnist

from public.data_loader_utils import load_csv

RESET = "\033[0m"
YELLOW = "\033[33m"


def load_mnist() -> pd.DataFrame:
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape images to rows of 784 pixels (28x28)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Convert to dataframes
    df_train = pd.DataFrame(X_train_flat)
    df_train["label"] = y_train
    # df_train['split'] = 'train'

    df_test = pd.DataFrame(X_test_flat)
    df_test["label"] = y_test
    # df_test['split'] = 'test'

    # Combine
    return pd.concat([df_train, df_test], ignore_index=True)


def base_report(df: pd.DataFrame, target: str) -> None:
    dict = {
        "rows \u00d7 cols": f"{df.shape[0]}\u00d7{df.shape[1]}",
        "n/a": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum(),
        "min/max": f"{df.drop(columns=[target]).values.min()}/{df.drop(columns=[target]).values.max()}",
        "0-columns": (df.drop(columns=[target]) == 0).all().sum(),
        "types": f"dtypes: {', '.join([f'{k}({v})' for k, v in df.dtypes.value_counts().items()])}",
    }
    display(pd.DataFrame([dict]).style.hide(axis="index"))


def get_reduced_mnist_data() -> pd.DataFrame:
    """
    Загрузка датасета
    """
    return load_csv(r"data\reduced.csv")


def _get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


@contextmanager
def memory_check():
    base_mb = _get_memory_usage()
    try:
        yield  # This runs the block of code inside the 'with' statement
    finally:
        new_mb = _get_memory_usage()
        print(f"{YELLOW}Memory Increased by: {new_mb - base_mb:.2f} MB{RESET}")


# Testing
if __name__ == "__main__":
    pass
