import numpy as np


def act(x):
    return 0 if x < 0.5 else 1

def go(money, toxic, location):
    x = np.array([money, toxic, location])
    DEEP_NEYRO_SL1_1 = [0.3, 0.3, 0]
    DEEP_NEYRO_SL1_2 = [0.8, 0.1, 0.4]

    # Двумерный массив (2 нейрона срытого слоя по 3 связи)
    WEIGHT_1 = np.array([DEEP_NEYRO_SL1_1, DEEP_NEYRO_SL1_2])

    # Вектор
    WEIGHT_2 = np.array([-1, 1])

    # Значение сумм скрытых нейронов
    sum_hidden = np.dot(WEIGHT_1, x)

    # Значение результата скрытых нейронов
    res_hidden = np.array([act(x) for x in sum_hidden])

    # Значение итогового нейрона
    res_end = np.dot(WEIGHT_2, res_hidden)
    y = act(res_end)

    return y

money = 1
toxic = 0
location = 1

if go(money, toxic, location) == 1:
    print('Пойдет')
else:
    print('Ищи новую работу')
