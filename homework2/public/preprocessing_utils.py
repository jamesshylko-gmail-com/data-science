import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ALLOWED_FILL_NA_STRATEGY: set[str] = {"mean", "median", "most_frequent"}


def drop_duplicates(df: pd.DataFrame, ignore_index: bool = True) -> pd.DataFrame:
    return df.drop_duplicates(df, ignore_index=ignore_index)


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Метод для удаления несуществунных колонок
    :param df: dataset
    :param strategy: стратегия заполнения
    :return: dataframe с замененными n/a
    """
    return df.drop(columns=columns)


def fill_missing_data(
    df: pd.DataFrame, strategy: str = "most_frequent"
) -> pd.DataFrame:
    """
    Метод для заполнения всех n/a позиций по заданной стратегии
    :param df: dataset
    :param strategy: стратегия заполнения
    :return: dataframe с замененными n/a
    """
    if strategy in ALLOWED_FILL_NA_STRATEGY:
        # Initialize SimpleImputer with the 'most_frequent' strategy
        imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)

        # Fit and transform the data
        # Note: SimpleImputer returns a NumPy array, so we wrap it back into a DataFrame
        return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    else:
        raise ValueError(f"Strategy {strategy} not supported yet")


# todo : Протестировать!
def one_hot_encoding(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Преобразует каждую строковую категорию в новый двоичный столбец ( 0 или 1).
    :param df: dataset
    :param columns: список имен столбцов для замены
    :return: исходный датасет, прирощенный новыми столбцпми
    """
    # 1. Initialize the encoder (sparse_output=False returns a standard NumPy array)
    encoder = OneHotEncoder(sparse_output=False)
    # 2. Fit and transform the data
    encoded_array = encoder.fit_transform(df[columns])
    # 3. Convert back to a DataFrame with proper column names
    encoded_df = pd.DataFrame(
        encoded_array, columns=encoder.get_feature_names_out(columns)
    )
    return df.update(encoded_df)


def ordinal_encoding(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Метод для заполнения всех не числовых колонок
    :param df: dataset
    :param columns: список названий столбцов для кодирования
    :return: dataframe с переведенными в числа строковыми данными по всем столбцам
    """
    # todo: not effective! Should be updated
    # 1. for every columns receive list values
    orders = {name: OrdinalEncoder().fit(df[[name]]).categories_[0] for name in columns}
    # 2. create encoder
    encoder = OrdinalEncoder(categories=list(orders.values()))

    # 3. Keep the columns matched exactly on both sides of the assignment
    df[columns] = encoder.fit_transform(df[columns])
    return df


# Как вернуть Total_Sales? И надо ли...
def standard_scaller(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    ВStandardScaler scikit-learn стандартизированы функции с помощьювычитание среднего значения и масштабирование до единичной дисперсииЭто означает, что ваши данные будут преобразованы таким образом, чтобы их среднее значение равнялось 0, а стандартное отклонение — 1
    :param df: dataset
    :return: dataframe с нормализованными данными
    """
    # 1. Initialize scaler and configure it to output Pandas DataFrames
    scaler = StandardScaler().set_output(transform="pandas")
    # 2. Fit to data and transform it
    return scaler.fit_transform(df)
