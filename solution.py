import pandas as pd
import numpy as np
from scipy.stats import t

chat_id = 361109448  # Ваш chat ID, не меняйте название переменной

def solution(sample_X: pd.Series, sample_Y: pd.Series) -> bool:
    # Вычисляем средние значения выборок
    mean_X = sample_X.mean()
    mean_Y = sample_Y.mean()

    # Вычисляем размеры выборок
    n_X = len(sample_X)
    n_Y = len(sample_Y)

    # Используем ошибки для оценки дисперсий
    variance_X = sample_X.std()**2
    variance_Y = sample_Y.std()**2

    # Вычисляем степени свободы
    degrees_of_freedom = n_X + n_Y - 2

    # Вычисляем t-статистику
    t_statistic = (mean_X - mean_Y) / np.sqrt((variance_X / n_X) + (variance_Y / n_Y))

    # Определяем критическое значение t для заданного уровня значимости (0.03)
    alpha = 0.03
    critical_value = t.ppf(1 - alpha / 2, degrees_of_freedom)

    # Определяем статистическую значимость различия
    reject_null_hypothesis = abs(t_statistic) > critical_value

    return reject_null_hypothesis
