import random

w1 = w2 = 1

def correctBias(x1, x2, b) :
    if logic_OR(x1, x2) and activationFunction(w1,w2,x1,x2,b) < 0.0:
        b+=0.5
        return float(b)
    if not logic_OR(x1, x2) and activationFunction(w1,w2,x1,x2,b) >= 0.0:
        b-=0.5
        return float(b)
    else:
        return b
    #if b == None:
    #    return 0.0

def activationFunction(w1, w2, x1, x2, b):
    return w1 * x1 + w2 * x2 + b

def isCorrect(x1,x2,b):
 if (logic_OR(x1,x2) and (activationFunction(w1,w2,x1,x2,b) > 0)) or (not logic_OR(x1,x2) and (activationFunction(w1,w2,x1,x2,b) < 0)):
     return True
 if (not logic_OR(x1,x2) and (activationFunction(w1,w2,x1,x2,b) > 0)) or (logic_OR(x1,x2) and (activationFunction(w1,w2,x1,x2,b) < 0)):
     return False
 else:
     return False

def logic_OR(x1,x2):
    if x1 + x2 > 0 :
        return True
    if x1 + x2 == 0:
        return False

def train_NN():
    b = 3
    for i in range(40):
        x1 = random.randint(0, 1)
        x2 = random.randint(0, 1)
        if (isCorrect(x1, x2, b)):
            print(x1, x2, b, activationFunction(w1, w2, x1, x2, b), isCorrect(x1, x2, b))
        else:
            print('Нужно изменить сдвиг', x1, x2, b)
            b = correctBias(x1, x2, b)
            print('Изменяем сдвиг', b)
    return b

def logic_OR_NN(x1,x2):
    if activationFunction(w1,w2,x1,x2,b) > 0:
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
b = train_NN()
print('Тест сети\n  ')
for data in test:
    print('При Х1 =', data[0], 'и Х2 =', data[1], 'Значение ИЛИ: ', logic_OR_NN(data[0], data[1]))
