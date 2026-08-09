import os
from contextlib import closing, contextmanager

import pandas as pd
import psutil
import psycopg2
import sqlalchemy
from IPython.display import display

from public.data_loader_utils import load_csv

RESET = "\033[0m"
YELLOW = "\033[33m"


def prepare_personality_data() -> pd.DataFrame:
    """
    Загрузка датасета
    """
    return load_csv(r"data\personality_synthetic_dataset.csv")


connection_config = {
    "host": "localhost",
    "database": "data_science",
    "user": "postgres",
    "password": "admin",
    "port": "5432",  # 5432 is the default PostgreSQL port
}


@contextmanager
def cursor():
    """
    Connect and get cursor
    commit and closing connection after
    """
    try:
        with closing(psycopg2.connect(**connection_config)) as conn:
            print("Successfully connected to the database!")
            with conn.cursor() as cursor:
                yield cursor
            conn.commit()
    except Exception as error:  # noqa: BLE001
        print(f"Error connecting to PostgreSQL: {error}")


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


def get_url():
    return sqlalchemy.URL.create(
        drivername="postgresql+psycopg2",
        username=connection_config.get("user"),
        password=connection_config.get("password"),
        host=connection_config.get("localhost"),
        port=int(connection_config.get("port")),
        database=connection_config.get("database"),
    )


# Testing
if __name__ == "__main__":
    df: pd.DataFrame = prepare_personality_data()
    # 2. Первичный осмотр
    print("Размер датасета (строки, столбцы):", df.shape)
    display(df.head())

    print("\nИнформация о типах данных и пропущенных значениях:")
    df.info()

    print("\nСтатистическая сводка числовых признаков:")
    display(df.describe())

    # 3. Проверка пропусков и дубликатов
    print("\nКоличество пропущенных значений в столбцах:")
    print(df.isnull().sum().sum())
    print("Количество дубликатов:", df.duplicated().sum())
