import pandas as pd

from public.data_loader_utils import load_csv


def prepare_blaban_data() -> pd.DataFrame:
    """
    Загрузка составляющих датасета и формирование единого 'стандартного' датасета
    """
    x_for_train = load_csv(r"data\data_blaban\blaban_train.csv")
    y_for_test = load_csv(r"data\data_blaban\blaban_test.csv")
    y_target_for_test = load_csv(r"data\data_blaban\blaban_submission.csv")

    y_for_test["Total_Sales"] = y_target_for_test["Predicted_Sales"]

    return pd.concat([x_for_train, y_for_test], axis=0)


def prepare_diabet_data() -> pd.DataFrame:
    """
    Загрузка датасета
    """
    return load_csv(r"data\data_diabet\diabetes.csv")


def prepare_taxi_data() -> pd.DataFrame:
    """
    Загрузка датасета
    """
    return load_csv(r"data\data_taxi\taxi_trip_pricing.csv")


def prepare_boston_data() -> pd.DataFrame:
    """
    Загрузка датасета
    """
    return load_csv(r"data\data_boston\boston.csv")


# Testing
if __name__ == "__main__":
    df: pd.DataFrame = prepare_blaban_data()
    print(df.head())
    print("-----------------------------------------------")
    print(df.info())
    print("-----------------------------------------------")
    print(df.dtypes)
