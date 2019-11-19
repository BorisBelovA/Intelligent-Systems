import random
import numpy as np

inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
expected_output = np.array([[0],[1],[1],[0]])

inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2,2,1
lr = 0.1
# Инициализация весов скрытого слоя разером 2*2
hiddenWeights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
hiddenBias = np.random.uniform(size=(1, hiddenLayerNeurons))
outputWeights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
outputBias = np.random.uniform(size=(1,outputLayerNeurons))
#print(hiddenWeights, hiddenBias)
#print('\n')
#print(outputWeights, outputBias)
#print('\n')


print("Initial hidden weights: ",end='')
print(*hiddenWeights)
print("Initial hidden biases: ",end='')
print(*hiddenBias)
print("Initial output weights: ",end='')
print(*outputWeights)
print("Initial output biases: ",end='')
print(*outputBias)

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

epochs=10000
for _ in range(epochs):
    hiddenLayerActivation = np.dot(inputs, hiddenWeights)
    #print(hiddenLayerActivation)
    #print('\n')
    hiddenLayerActivation += hiddenBias
    #print(hiddenLayerActivation)
    #print('\n')

    hiddenLayerOutput = sigmoid(hiddenLayerActivation)
    #print(hiddenLayerOutput)

    outputLayerActivation = np.dot(hiddenLayerOutput, outputWeights)
    outputLayerActivation += outputBias
    predictedOutput = sigmoid(outputLayerActivation)

    #BackPropagation
    error = expected_output - predictedOutput
    d_predicted_output = error * sigmoid_derivative(predictedOutput)
    #print(error, end="\n")
    #print(d_predicted_output, end="\n")
    error_hidden_layer = d_predicted_output.dot(outputWeights.T)
    #print(error_hidden_layer, end="\n")
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hiddenLayerOutput)
    #print(d_hidden_layer, end="\n")

    outputWeights += hiddenLayerOutput.T.dot(d_predicted_output) * lr
    outputBias += np.sum(d_predicted_output, axis=0, keepdims=True)
    hiddenWeights += inputs.T.dot(d_hidden_layer) * lr
    hiddenBias += np.sum(d_hidden_layer, axis=0, keepdims=True)

print("Final hidden weights: ",end='')
print(*hiddenWeights)
print("Final hidden bias: ",end='')
print(*hiddenBias)
print("Final output weights: ",end='')
print(*outputWeights)
print("Final output bias: ",end='')
print(*outputBias)

print("\nOutput from neural network after 10,000 epochs: ",end='')
print(*predictedOutput)