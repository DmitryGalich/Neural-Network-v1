from Lessons.functions import init, run, back_propagation, table
from Lessons.functions import output, input_0, input_1, input_2
from Lessons.functions import Max_allowed_error

variations_table = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

for i in range(10):
    for variant in variations_table:
        init(variant[0], variant[1], variant[2])
        run()
        while abs(output.error) > Max_allowed_error:
            back_propagation()
            run()
            print("Вход: [%i,%i,%i] Максимальная допустимая ошибка: %f Ошбика: %f"
                  % (variant[0], variant[1], variant[2], Max_allowed_error, output.error))


table()


while variations_table[0] == [0, 0, 0]:
    if input("Проверить?") == "stop":
        break
    else:
        input_0.value = float(input("input 0: "))
        input_1.value = float(input("input 1: "))
        input_2.value = float(input("input 2: "))

        run()
        print(output.value)







