import random

w1 = 1


def activation_function(w1,x1,b):
    return w1 * x1 + b


def login_not(x1):
    if x1 > 0:
        return False
    else:
        return True


def is_correct(x1,b):
    if (not login_not(x1) and (activation_function(w1,x1,b) > 1)) or (login_not(x1) and (activation_function(w1,x1,b) <= 1)):
        return True
    if(not login_not(x1) and (activation_function(w1,x1,b) <= 1)) or (login_not(x1) and (activation_function(w1,x1,b) > 1)):
        return False
    else:
        return False


def correct_bias(x1,b):
    if not login_not(x1) and (activation_function(w1,x1,b) <= 1):
        b += 0.5
        return b
    if login_not(x1) and (activation_function(w1,x1,b) > 1):
        b -= 0.5
        return b
    return b


def train_nn():
    b = 3
    for i in range(15):
        x1 = random.randint(0,1)
        if is_correct(x1,b):
            print(x1, b, activation_function(w1, x1, b), is_correct(x1, b))
        else:
            print('Нужно изменить сдвиг', x1, b)
            b = correct_bias(x1, b)
            print('Изменяем сдвиг', b)
    return b


def logic_not_nn(x1):
    if activation_function(w1, x1, b) > 1:
        return 0
    else:
        return 1

b = train_nn()

test = [
    0,
    1
]
print('Обучение сети\n')
b = train_nn()
print('Тест сети\n  ')
for data in test:
    print('При Х1 =', data, 'Значение НЕ: ', logic_not_nn(data))