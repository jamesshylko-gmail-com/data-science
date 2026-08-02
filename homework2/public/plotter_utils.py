import pandas as pd
from matplotlib import pyplot as plt


def draw_dot_plot(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
):
    num_points = len(y_true)
    # Визуализация matplotlib
    plt.figure(figsize=(10, 6))
    plt.scatter(range(num_points), y_true, color="blue", label="Истинные значения")
    plt.scatter(range(num_points), y_pred, color="red", label="Предсказанные значения")
    plt.xlabel("Наблюдения")
    plt.ylabel("Значения")
    plt.title(
        f"Сравнение истинных и предсказанных значений (первые {num_points} точек)"
    )
    plt.legend()
    plt.grid(True)
    plt.show()
