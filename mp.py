#Author: Monynich Kiem
#Date: 03/08/2022
#Desc: CS460G Multilayer Perceptrons
#using MNIST dataset
#each training example consists of csvs with 785 features, the first is a numeric label and the rest are image pixels
#can have single or multiple output nodes
#needs at least one hidden layer and nodes, the more layers and nodes: the more accurate
#activation function is the sigmoid!
#need to include accuracy
#for this, the code is a modified version of Dr. Harrison's pseudocode given in class

import numpy as np
import pandas as pd
#import warnings
import difflib
#thanks to Thomas Dimeny for showing me this useful library!
#warnings.filterwarnings('ignore')

#hyperparameters
alpha = .001
epoch = 10

#activation function
def sigmoid(exponent):
    return 1/(1 + np.exp(-exponent))
#print(sigmoid(2))
#it works great moving on

#reading in dataframe
train_data = pd.read_csv('mnist_train_0_1.csv', header = None)
test_data = pd.read_csv('mnist_test_0_1.csv', header = None)

#print(train_data.shape) #12665x785
#print(test_data.shape) #2115x785

#normalize the features b/c some are really large numbers
def normalize(dataset):
    normalized = dataset/255
    return normalized

#begin train:
labels = []
outputs = []

hidden_weights = []
output_weights = []
num_nodes = 3

#785 features
for i in range(785):
    hidden_weights.append(list(np.random.uniform(-1,1,num_nodes)))

#number of hidden nodes
for i in range(num_nodes):
    output_weights.append(np.random.uniform(-1,1))

#take from linear regression proj and pseudocode
for data in range(12665):
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
        
        #updating weights
        temp_output = output_weights
        for i in range(num_nodes):
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
for data in range(2115):
    test_input = test_data.iloc[data]
    test_label = test_input[0]
    test_labels.append(test_label)
    test_input = test_input.to_numpy()
    test_input[0] = 1
    test_input = normalize(test_input)
    
    #only need forward pass
    hidden_input = np.dot(list(np.array(final_hidden).transpose()), test_input)
    hidden_output = list(map(sigmoid, hidden_input))
    
    overall_output = sigmoid(np.dot(final_output, hidden_output))
    overall_output = round(overall_output)
    
    predictions.append(overall_output)

#https://stackoverflow.com/questions/12436672/how-does-sequencematcher-ratio-works-in-difflib
sm=difflib.SequenceMatcher(None, test_labels, predictions, autojunk = False)
print(sm.ratio()*100)

