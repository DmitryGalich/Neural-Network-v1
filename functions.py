import math
from Lessons.classes import Input, Hidden, Output

# Инициализация классов
input_0 = Input()
input_1 = Input()
input_2 = Input()
hidden_0 = Hidden()
hidden_1 = Hidden()
output = Output

Expectations = 0.0
Learning_rate = float(input("Коэффициент точности обучения: "))
Max_allowed_error = float(input("Максимаьная допустимая ошибка: "))


def init(First, Second, Third):
    global Expectations

    input_0.get_value(First)
    input_1.get_value(Second)
    input_2.get_value(Third)

    hidden_0.error = hidden_1.error = output.error = 0.0

    Expectations = expectations(input_0.value, input_1.value, input_2.value)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_dx(x):
    return sigmoid(x) * (1 - sigmoid(x))


def expectations(first, second, third):
    if (first == 0 and second == 0 and third == 0 or
            first == 0 and second == 1 and third == 0 or
            first == 1 and second == 0 and third == 0 or
            first == 1 and second == 1 and third == 0):
        return 0
    else:
        return 1


def get_hidden_value(value_0, weight_0, value_1, weight_1, value_2, weight_2):
    return sigmoid(value_0 * weight_0 + value_1 * weight_1 + value_2 * weight_2)


def get_output_value(value_0, weight_0, value_1, weight_1):
    return sigmoid(value_0 * weight_0 + value_1 * weight_1)


def refresh_weight(weight, value, weight_delta):
    return weight - value * weight_delta * Learning_rate


def run():
    hidden_0.value = get_hidden_value(input_0.value, input_0.weight_to_next_0,
                                      input_1.value, input_1.weight_to_next_0,
                                      input_2.value, input_2.weight_to_next_0)
    hidden_1.value = get_hidden_value(input_0.value, input_0.weight_to_next_1,
                                      input_1.value, input_1.weight_to_next_1,
                                      input_2.value, input_2.weight_to_next_1)
    output.value = get_output_value(hidden_0.value, hidden_0.weight_to_next,
                                    hidden_1.value, hidden_1.weight_to_next)
    output.get_error(Output, Expectations)


def back_propagation():
    output.get_weight_delta(Output, sigmoid_dx(output.value))
    hidden_0.weight_to_next = refresh_weight(hidden_0.weight_to_next, hidden_0.value, output.weight_delta)
    hidden_1.weight_to_next = refresh_weight(hidden_1.weight_to_next, hidden_1.value, output.weight_delta)

    hidden_0.get_error(output.weight_delta)
    hidden_0.get_weight_delta(sigmoid_dx(hidden_0.value))
    hidden_1.get_error(output.weight_delta)
    hidden_1.get_weight_delta(sigmoid_dx(hidden_1.value))

    input_0.weight_to_next_0 = refresh_weight(input_0.weight_to_next_0, input_0.value, hidden_0.weight_delta)
    input_1.weight_to_next_0 = refresh_weight(input_1.weight_to_next_0, input_1.value, hidden_0.weight_delta)
    input_2.weight_to_next_0 = refresh_weight(input_2.weight_to_next_0, input_2.value, hidden_0.weight_delta)

    input_0.weight_to_next_1 = refresh_weight(input_0.weight_to_next_1, input_0.value, hidden_1.weight_delta)
    input_1.weight_to_next_1 = refresh_weight(input_1.weight_to_next_1, input_1.value, hidden_1.weight_delta)
    input_2.weight_to_next_1 = refresh_weight(input_2.weight_to_next_1, input_2.value, hidden_1.weight_delta)


def table():
    print("\nЗначение input_0: %f" % input_0.value)
    print("Вес input_0 => hidden_0: %f" % input_0.weight_to_next_0)
    print("Вес input_0 => hidden_1: %f" % input_0.weight_to_next_1)

    print("\nЗначение input_1: %f" % input_1.value)
    print("Вес input_1 => hidden_0: %f" % input_1.weight_to_next_0)
    print("Вес input_1 => hidden_1: %f" % input_1.weight_to_next_1)

    print("\nЗначение input_2: %f" % input_2.value)
    print("Вес input_2 => hidden_0: %f" % input_2.weight_to_next_0)
    print("Вес input_2 => hidden_1: %f" % input_2.weight_to_next_1)

    print("\nЗначение hidden_0: %f" % hidden_0.value)
    print("Вес hidden_0 => output: %f" % hidden_0.weight_to_next)

    print("\nЗначение hidden_1: %f" % hidden_1.value)
    print("Вес hidden_1 => output: %f" % hidden_1.weight_to_next)

    print("\nЗначение output: %f" % output.value)
    print("\nОжидания: %f" % Expectations)
    print("\nОшибка: %f" % output.error)
