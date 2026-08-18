from typing import Self

import pandas as pd

import public.classification_utils as cla


class ModelNotFoundException(Exception):
    pass


class ClassificationProcessor:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.original = df
        self.target_column = target_column
        self.models = []

    def calculate(self, function_list: dict[str, dict]) -> Self:
        """
        train + predict + evaluate + prepare data for report
        """
        for key in function_list:
            func = getattr(cla, key)
            self.models.append(
                func(
                    df=self.original.copy(),
                    target_column=self.target_column,
                    opts=function_list.get(key),
                )
            )
        return self

    def report(self, title: str):
        model = next((item for item in self.models if item.title == title), None)
        if not model:
            raise ModelNotFoundException(f"Model for '{title}' not found")
        cla.report(model)

    def pick_model(self, title: str) -> None:
        model = next((item for item in self.models if item.title == title), None)
        if not model:
            raise ModelNotFoundException(f"Model for '{title}' not found")
        cla.pick(model, title)

    # todo : outsource to report_utils
    # def report(self) -> None:
    #     for value in self.models.values():
    #         cla.report(value)
