import pandas as pd


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Загрузка данных из CSV файла.
    :param file_path: Путь к CSV файлу.
    :return: DataFrame с загруженными данными.
    """
    with open(file_path, mode="r", encoding="utf-8") as file:
        return pd.read_csv(file)


def load_json():
    pass


def load_api():
    pass


# Testing
if __name__ == "__main__":
    data = load_csv(r"data\test\data.csv")
    print(data)
