import random

num0 = list('111101101101111')
num1 = list('010010010010010')
num2 = list('111001111100111')
num3 = list('111001111001111')
num4 = list('101101111001001')
num5 = list('111100111001111')
num6 = list('111100111101111')
num7 = list('111001001001001')
num8 = list('111101111101111')
num9 = list('111101111001111')

numbers = [num0, num1, num2, num3, num4, num5, num6, num7, num8, num9]


num11 = list('010010010010010')
num12 = list('010110010010010')
num13 = list('010110010010111')
num14 = list('010010010010111')


# Виды цифры 5 (Тестовая выборка)
num51 = list('111100111000111')
num52 = list('111100010001111')
num53 = list('111100011001111')
num54 = list('110100111001111')
num55 = list('110100111001011')
num56 = list('111100101001111')

# Веса нейронов
weights = [i for i in range(15)]


#Порог
bias = 7

def proceed(number):
    sum = 0
    for i in range(len(number)):
        sum += weights[i]*int(number[i])
    return sum >= bias

# Уменьшаем вес, если сеть ошиблась и выдала 1
def decrease(number):
    for i in range(len(number)):
        if(int(number[i]) == 1):
            weights[i] -= 1

def increase(number):
    for i in range(len(number)):
        if(int(number[i])==1):
            weights[i] += 1

def train(epochs):
    for i in range(epochs):
        option = random.randint(0, 9)

        if (option != numberToRecognize):
            if (proceed(numbers[option])):
                decrease(numbers[option])
        else:
            if not proceed(numbers[option]):
                increase(numbers[option])



def printReults() :
    print('Веса: ', weights, '\n')

    # Прогон по обучающей выборке
    print("0 это ", numberToRecognize, "? ", proceed(num0))
    print("1 это ", numberToRecognize, "? ", proceed(num1))
    print("2 это ", numberToRecognize, "? ", proceed(num2))
    print("3 это ", numberToRecognize, "? ", proceed(num3))
    print("4 это ", numberToRecognize, "? ", proceed(num4))
    print("6 это ", numberToRecognize, "? ", proceed(num6))
    print("7 это ", numberToRecognize, "? ", proceed(num7))
    print("8 это ", numberToRecognize, "? ", proceed(num8))
    print("9 это ", numberToRecognize, "? ", proceed(num9), '\n')

    # Прогон по тестовой выборке
    print("Узнал ", numberToRecognize, "? ", proceed(num1))
    print("Узнал ", numberToRecognize, "? ", "- 1? ", proceed(num11))
    print("Узнал ", numberToRecognize, "? ", "- 2? ", proceed(num12))
    print("Узнал ", numberToRecognize, "? ", "- 3? ", proceed(num13))
    print("Узнал ", numberToRecognize, "? ", "- 4? ", proceed(num14))
    #print("Узнал ", numberToRecognize, "? ", " - 5? ", proceed(num55))
    #print("Узнал ", numberToRecognize, "? ", " - 1? ", proceed(num56))

numberToRecognize = 1

train(25000)
printReults()
