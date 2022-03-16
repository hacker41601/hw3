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
import math

#hyperparameters
alpha = .001
epoch = 42

#activation function
def sigmoid(exponent):
    sigmoid = 1 / (1 + np.exp(-exponent))
    return sigmoid
#print(sigmoid(2)) #it works great moving on

#reading in dataframes
train01 = pd.read_csv('mnist_train_0_1.csv')
test01 = pd.read_csv('mnist_test_0_1.csv')
#train04 = pd.read_csv('mnist_train_0_4.csv')
#test04 = pd.read_csv('mnist_test_0_4.csv')

#print(train01)
#print(train01.drop(train01.columns[[0]], axis = 1))

#normalize the features after column 1 b/c the first column is the labels
def normalize(dataset):
    norm_dataset = dataset.copy(deep = True)
    norm_dataset = norm_dataset.drop(norm_dataset.columns[[0]], axis = 1)
    norm_dataset/255
    return round(norm_dataset)

train_feat = normalize(train01)
#print(train_feat)
train_label = train01.iloc[:,0]
#print(train_cat)

test_feat = normalize(test01)
#print(test_feat)
test_label = test01.iloc[:,0]
#print(test_cat)

train_feat = np.array(train_feat)
train_label = np.array(train_label)

test_feat = np.array(test_feat)
test_label = np.array(test_label)

weights = []
bias = []

for i in range(2):
    if i == 0: #first layer aka input layer
        weights.append(np.random.uniform(-1, 1, (len(test_feat[i]), 2)))
        bias.append(np.random.uniform(-1, 1, (2,1)))
    elif i != 1: #hidden layer
        weights.append(np.random.uniform(-1, 1, (2, 2)))
        bias.append(np.random.uniform(-1, 1, (2,1)))
    else: #output aka last layer
        weights.append(np.random.uniform(-1, 1, (2,1)))
        bias.append(np.random.uniform(-1, 1, (1,1)))
    
    print(str(i))
    print(type(bias[i]))
    print(bias[i].shape)
    print(weights[i].shape)

for curr_epoch in range(epoch):
    for i, ex in enumerate(train_feat): #enumerate loops over an interable object and keeps track of how many iterations have occured
        #begin forward pass
        features = train_feat[i]
        #print(features.shape)
        ground_truth = float(train_label[i])
        #print(ground_truth)
        output = []
        for j in range(2):
            if j == 0:
                prod = np.transpose(weights[j]).dot(features)
            else:
                prod = np.transpose(weights[j]).dot(output[j-1])
            input = prod + np.transpose(bias[j])
            input = np.transpose(input)
            input = np.array(input)
            sigmoid(input)
            output.append(input)
            #print(output)
            #print(output[j].shape)
        output = float(output[-1])
        #print(output)
        raw_error = ground_truth - output
        #print(error)
        #not sure if the logic is there will ask Thomas/Dr. Harrison
        
        #help here
        #backprop stuff
        deltas = []
        for k in reversed(range(2)):
            if k == 1:
                delta = raw_error * (output * (1-output)) #scalar operation
            else:
                delta = weights[k+1].dot(deltas[0]) * (output[k] * (1-output[k]))
                delta = np.array(delta)
            deltas.insert(0, delta)
            print(deltas[0])
