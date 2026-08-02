from contextlib import contextmanager
from typing import Self

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import public.preprocessing_utils as pre
from private import utils
from public.log_utils import log_time
from public.plotter_utils import draw_dot_plot


class LinearRegressionModel:
    STATUSES: tuple[str] = (
        "UNDEFINED",
        "INIT",
        "PREPROCESSING",
        "TRAIN",
        "PREDICT",
        "DONE",
    )
    current_status: int = 0

    @contextmanager
    def status_context(self, status_name: str | None = None):
        if not (
            (not status_name and not self.current_status)
            or self.STATUSES[self.current_status] == status_name
        ):
            raise ValueError(
                f"Нарушена последовательность действий: не достигнут статус '{status_name}'"
            )
        try:
            yield  # This runs the block of code inside the 'with' statement
        finally:
            if self.STATUSES[self.current_status] != "DONE":
                self.current_status += 1

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.3,
        split_type: str = "not suffle",
    ):
        with self.status_context():
            if split_type == "not suffle":
                self.target_column = target_column
                train_size = int(len(df) * (1 - test_size))
                self.df_train = df.iloc[:train_size]
                self.df_test = df.iloc[train_size:]
            else:
                raise ValueError(f"Split type {split_type} not supported")

    @log_time
    def preprocessing(self, function_list: dict[str, dict]) -> Self:
        with self.status_context("INIT"):
            # for func in function_list:
            for key in function_list:
                func = getattr(pre, key)

                self.df_train = func(self.df_train, **function_list.get(key))
                self.df_test = func(self.df_test, **function_list.get(key))
            return self

    @log_time
    def training(self) -> Self:
        with self.status_context("PREPROCESSING"):
            self.model = LinearRegression().fit(
                self.df_train.drop(columns=[self.target_column]),
                self.df_train[self.target_column],
            )
            return self

    @log_time
    def predict(self) -> Self:
        with self.status_context("TRAIN"):
            self.y_predicted = pd.DataFrame(
                self.model.predict(self.df_test.drop(columns=[self.target_column])),
                columns=[self.target_column],
            )
            return self

    @log_time
    def evaluate(self) -> Self:
        with self.status_context("PREDICT"):
            y_true = self.df_test[self.target_column]
            y_pred = self.y_predicted[self.target_column]
            self.mean_squared_error = ((y_true - y_pred) ** 2).mean()
            self.r2 = r2_score(y_true, y_pred)
            return self

    @log_time
    def report(self) -> None:
        with self.status_context("DONE"):
            print(
                f"Коэффициенты линейной регрессии:\n\t{self.model.coef_}\nIntercept:\n\t{self.model.intercept_}"
            )
            print(f"Среднеквадратичная ошибка:\n\t{self.mean_squared_error}")
            print(f"Коэффициент детерминации R^2:\n\t{self.r2}")

    def draw(self, num_points=50) -> None:
        with self.status_context("DONE"):
            y_true = self.df_test[self.target_column][:num_points]
            y_pred = self.y_predicted[self.target_column][:num_points]
            draw_dot_plot(y_true=y_true, y_pred=y_pred)


# Testing
if __name__ == "__main__":
    data_frame: pd.DataFrame = utils.prepare_blaban_data()
    model: LinearRegressionModel = (
        LinearRegressionModel(data_frame)  # set datadrame
        .preprocessing(  # transmrm dataframe based on analizer info
            {
                # "drop_duplicates": {},
                "drop_columns": {"columns": ["Transaction_ID", "Date_Time"]},
                "fill_missing_data": {"strategy": "most_frequent"},
                # 'one_hot_encoding' : {"columns": ["Customer_Gender"]},
                "ordinal_encoding": {
                    "columns": [
                        "Branch",
                        "Product_Name",
                        "Size",
                        "Topping_Type",
                        "Customer_Gender",
                        "Membership_Status",
                        "Payment_Method",
                        "Order_Source",
                        "Category",
                        "Region",
                    ]
                },
                "standard_scaller": {},
            }
        )
        .training()  # trainig
        .predict()  # predicttion based on test data
        .evaluate()  # evaluate model presicion
    )
    # report(model.df_test, [dataset_stats_info, missing_value_info])
    model.report()
    model.draw()
