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
#originally had epoch as 42, but that took several minutes to compute
#tried larger alphas but ended up in overflow errors

#hyperparameters
alpha = .001
epoch = 10

#activation function using sigmoid equation but with numpy functions
def sigmoid(exponent):
    return 1/(1 + np.exp(-exponent))
#print(sigmoid(2))
#it works great moving on

#normalize the features b/c some are really large numbers
def normalize(dataset):
    normalized = dataset/255
    return normalized

#reading in dataframes
train_data = pd.read_csv('mnist_train_0_1.csv', header = None)
test_data = pd.read_csv('mnist_test_0_1.csv', header = None)
#print(train_data.shape) #12665x785
#print(test_data.shape) #2115x785

#begin train:
#initialize arrays for labels, outputs, hidden weights, and output weights, as well as number of nodes
labels = []
outputs = []
hidden_weights = [] #785 x 3
output_weights = [] #3 x 1
num_nodes = 3
num_feat = len(train_data.columns)
#print(num_feat)

#785 features
for i in range(num_feat):
    hidden_weights.append(list(np.random.uniform(-1,1,num_nodes)))

#number of hidden nodes
for i in range(num_nodes):
    output_weights.append(np.random.uniform(-1,1))

#take from linear regression proj and pseudocode
for data in range(len(train_data)):
    #separate labels and inputs, replace labels with bias column!!!
    input = train_data.iloc[data]
    label = input[0]
    labels.append(label)
    #labels.append(input[0]) this messes it all up for some reason
    input = input.to_numpy()
    #replace first column with bias of ones rather than inserting ANOTHER column since the labels are already stored in the array
    input[0] = 1
    input = normalize(input) #normalize the data or else it gets all wonky
    curr_epoch = 0
    while curr_epoch <= epoch:
    #begin forward pass from Dr. Harrison's code
        hidden_input = np.dot(list(np.array(hidden_weights).transpose()), input)
        hidden_output = list(map(sigmoid, hidden_input))
        overall_output = sigmoid(np.dot(output_weights, hidden_output))
        
    #begin back prop from Dr. Harrison's code
        delta_output = (label - overall_output) * (overall_output) * (1 - overall_output) #sigmoid prime is ov out * (1- ov out)
        delta_hidden = np.dot(delta_output, output_weights)
        gradient_hidden = np.transpose(np.outer(delta_hidden, input))
        #gradient_hidden = np.transpose(gradient_hidden)
        
        #updating weights similar to linear regression
        temp_output = output_weights
        for i in range(num_nodes):
            output_weights[i] = temp_output[i] + alpha * hidden_output[i] * delta_output
        temp_hidden = hidden_weights
        for i in range(num_feat):
            hidden_weights[i] = temp_hidden[i] + alpha * input[i] * gradient_hidden[i]
        
        overall_output = round(overall_output)
        #print(overall_output)
        curr_epoch += 1
            
    final_hidden = hidden_weights
    final_output = output_weights

#begin test
test_labels = []
predictions = []
for data in range(len(test_data)):
    test_input = test_data.iloc[data]
    test_label = test_input[0]
    test_labels.append(test_label)
    #test_labels.append(input[0]) this messes it all up for some reason
    test_input = test_input.to_numpy()
    test_input[0] = 1
    test_input = normalize(test_input)
    
    #only need forward pass using the train data!!!
    hidden_input = np.dot(list(np.array(final_hidden).transpose()), test_input)
    hidden_output = list(map(sigmoid, hidden_input))
    overall_output = sigmoid(np.dot(final_output, hidden_output))
    overall_output = round(overall_output)
    #print(overall_output)
    predictions.append(overall_output)

#https://stackoverflow.com/questions/12436672/how-does-sequencematcher-ratio-works-in-difflib
sm=difflib.SequenceMatcher(None, test_labels, predictions, autojunk = False)
print("Accuracy: ")
print(sm.ratio()*100)
print(" ")
print("Error: ")
print(100 - (sm.ratio()*100))
