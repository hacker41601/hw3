import numpy as np
import math
import random

def sigmoid(x):
    return 1/(1+math.pow(math.e, -x))

input = [1,2,3,4] #4x1
output = [-1]

weights_hidden = [] #4x3
weights_output = [] #3x1

for i in range(len(input)):
    weights_hidden(list(np.random.uniform(-1,1,3)))
    
for j in range(3):
    weights_output.append(np.random.uniform(-1,1))
    
print(weights_output)
print(weights_hidden)

#forward pass:
#generate hidden outputs

hidden_in = np.dot(list(np.array(weights_hidden).transpose()), input)
hidden_out = list(map(sigmoid, hidden_in))

#generate full output
full_output = sigmoid(np.dot(weights_output, hidden_out))
print(full_output)

#forward pass complete

#back prop
#calculate deltas

delta_output = 0.1
delta_hidden = np.dot(delta_output, weights_output)
print(delta_hidden)

gradients_hidden = np.outer(delta_hidden, input)
print(delta_hidden)
print(input)
print(np.transpose(np.array(weights_hidden)))
print(gradients_hidden)
