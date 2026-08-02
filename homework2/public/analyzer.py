import textwrap
from collections.abc import Callable

import pandas as pd

from private import utils

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"


def report(df: pd.DataFrame, function_list: list[Callable]) -> None:
    """
    Метод предназначен для получения предварительной информации по датасету

    :param  df: 'сырой' датасет
    :param function_list: список функций (в нужном для отчета порядке), формирующих данные для отчета
    :return: None
    """
    result: dict = {}
    for func in function_list:
        result.update(func(df))

    for key, value in result.items():
        print(f"\n{GREEN}{BOLD}{key}{RESET}")
        if isinstance(value, list):
            print(textwrap.indent("\n".join(value), "\t"))
        elif isinstance(value, (tuple, set, pd.Series, pd.DataFrame)):
            print(
                textwrap.indent(
                    value.__str__() if not isinstance(value, str) else value, "\t"
                )
            )
        else:
            print(f"\t{value}")


def dataset_stats_info(df: pd.DataFrame) -> dict:
    """
    Информация о размере выборки, несколько строк датасета

    :param  df: 'сырой' датасет
    :return: словарь инфформации для отчета
    """
    return {"total": len(df), "head": df.head()}


def duplicates_info(df: pd.DataFrame) -> dict:
    """
    Информация о наличии дубликатов в датасете

    :param  df: 'сырой' датасет
    :return: словарь инфформации для отчета
    """
    duplicates: pd.DataFrame = df[df.duplicated(keep=False)]
    return {
        "total duplicates": len(duplicates),
        "duplicates": duplicates.head() if len(duplicates) else None,
    }


def missing_value_info(df: pd.DataFrame) -> dict:
    """
    Информация о наличии пустых значений в датасете

    :param  df: 'сырой' датасет
    :return: словарь инфформации для отчета
    """
    return {"columns with missed values": df.isna().sum().loc[lambda x: x > 0]}


def not_numeric_columns_info(df: pd.DataFrame) -> dict:
    """
    Информация о наличии не числовых колонок в датасете

    :param  df: 'сырой' датасет
    :return: словарь инфформации для отчета
    """
    not_numeric_columns_with_examples: list[str] = []

    for col, dtype in df.select_dtypes(exclude=["int64", "float64"]).dtypes.items():
        val = df[col].iloc[0]
        not_numeric_columns_with_examples.append(f"{col:<20} {dtype!s:<10} {val!s:<20}")
    return {"columns with not numeric types": not_numeric_columns_with_examples}


def anomalies_info(df: pd.DataFrame) -> dict:
    """
    Информация о наличии аномалий в датасете

    :param  df: 'сырой' датасет
    :return: словарь инфформации для отчета
    """
    return {"anomalies": "Not implemented yet"}


# Testing
if __name__ == "__main__":
    data_frame: pd.DataFrame = utils.prepare_blaban_data()
    report(
        data_frame,
        [
            dataset_stats_info,
            duplicates_info,
            missing_value_info,
            not_numeric_columns_info,
            anomalies_info,
        ],
    )
