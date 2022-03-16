#Author: Monynich Kiem
#Date: 03/08/2022
#Desc: CS460G Multilayer Perceptrons
#using MNIST dataset
#each training example consists of csvs with 785 features, the first is a numeric label and the rest are image pixels
#can have single or multiple output nodes
#needs at least one hidden layer and nodes, the more layers and nodes: the more accurate
#activation function is the sigmoid!
#need to include accuracy
#Author: Monynich Kiem
#Date: 03/08/2022
#Desc: CS460G Multilayer Perceptrons
#using MNIST dataset
#each training example consists of csvs with 785 features, the first is a numeric label and the rest are image pixels
#can have single or multiple output nodes
#needs at least one hidden layer and nodes, the more layers and nodes: the more accurate
#activation function is the sigmoid!
#need to include accuracy

import numpy as np
import pandas as pd
import difflib

#hyperparameters
alpha = .1
epoch = 42

#activation function
def sigmoid(exponent):
    return 1/(1 + np.exp(-exponent))
#print(sigmoid(2)) #it works great moving on

#reading in dataframe
train_data = pd.read_csv('mnist_train_0_1.csv', header = None)
test_data = pd.read_csv('mnist_test_0_1.csv', header = None)

#normalize the features after column 1 b/c the first column is the labels
def normalize(dataset):
    normalized = dataset/255
    return normalized

#begin train:
labels = []
outputs = []

hidden_weights = []
output_weights = []

#785 features
for i in range(785):
    hidden_weights.append(list(np.random.uniform(-1,1,4)))

#number of hidden nodes
for i in range(4):
    output_weights.append(np.random.uniform(-1,1))

#take from linear regression proj
for data in range(400):
    input = train_data.iloc[data]
    label = input[0]
    labels.append(label)
    input = input.to_numpy()
    input[0] = 1
    input = normalize(input)
    curr_epoch = 0
    while curr_epoch <= epoch:
    #begin forward pass
        hidden_input = np.dot(list(np.array(hidden_weights).transpose()), input)
        hidden_output = list(map(sigmoid, hidden_input))
        overall_output = sigmoid(np.dot(output_weights, hidden_output))
        
    #begin back prop
        delta_output = (label - overall_output) * (overall_output) * (1 - overall_output)
        delta_hidden = np.dot(delta_output, output_weights)
        gradient_hidden = np.outer(delta_hidden, input)
        gradient_hidden = np.transpose(gradient_hidden)
        
        temp_output = output_weights
        for i in range(4):
            output_weights[i] = temp_output[i] + alpha * hidden_output[i] * delta_output
            
        temp_hidden = hidden_weights
        for i in range(785):
            hidden_weights[i] = temp_hidden[i] + alpha * input[i] * gradient_hidden[i]
        
        overall_output = round(overall_output)
        #print(overall_output)
        curr_epoch += 1
        if curr_epoch == epoch:
            outputs.append(overall_output)
    final_hidden = hidden_weights
    final_output = output_weights

#begin test
test_labels = []
predictions = []
for data in range(1000):
    test_input = test_data.iloc[data]
    test_label = test_input[0]
    test_labels.append(test_label)
    test_input = test_input.to_numpy()
    test_input[0] = 1
    test_input = normalize(test_input)
    
    hidden_input = np.dot(list(np.array(final_hidden).transpose()), test_input)
    hidden_output = list(map(sigmoid, hidden_input))
    
    overall_output = sigmoid(np.dot(final_output, hidden_output))
    overall_output = round(overall_output)
    
    predictions.append(overall_output)
    
sm=difflib.SequenceMatcher(None, test_labels, predictions, autojunk = False)
    
print(sm.ratio()*100)

