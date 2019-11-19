import random

w1 = w2 = 1


def activation_function(w1, w2, x1, x2, b):
    return w1 * x1 + w2 * x2 + b


def logic_and(x1, x2):
    if(x1 + x2 > 1):
        return True
    else:
        return False


def is_correct(w1,w2, x1,x2,b):
    if (logic_and(x1,x2) and (activation_function(w1,w2,x1,x2,b) > 1)) or (not logic_and(x1,x2) and (activation_function(w1,w2,x1,x2,b) <= 1)):
        return True
    if (logic_and(x1,x2) and (activation_function(w1,w2,x1,x2,b) <= 1)) or (not logic_and(x1,x2) and (activation_function(w1,w2,x1,x2,b) > 1)):
        return False
    else:
        return False


def correct_bias(w1,w2,x1,x2,b):
    if logic_and(x1,x2) and activation_function(w1,w2,x1,x2,b) <= 1:
        b += 0.5
        return b
    if not logic_and(x1,x2) and activation_function(w1,w2,x1,x2,b) > 1:
        b -= 0.5
        return b
    return b


def train_nn():
    b = 3
    for i in range(40):
        x1 = random.randint(0,1)
        x2 = random.randint(0,1)
        if is_correct(w1,w2,x1,x2,b):
            print(x1, x2, b, activation_function(w1, w2, x1, x2, b), is_correct(w1,w2,x1,x2,b))
        else:
            print('Нужно изменить сдвиг', x1, x2, b)
            b = correct_bias(w1,w2,x1,x2,b)
            print('Изменяем сдвиг', b)
    return b


def logic_and_nn(x1,x2):
    if activation_function(w1,w2,x1,x2,b) > 1:
        return 1
    else:
        return 0

test = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]
print('Обучение сети\n')
b = train_nn()
print('Тест сети\n  ')
for data in test:
    print('При Х1 =', data[0], 'и Х2 =', data[1], 'Значение И: ', logic_and_nn(data[0], data[1]))

