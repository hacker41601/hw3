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
hidden_layers = 1
hidden_nodes = 2
output_nodes = 1
total_layers = hidden_layers + 1 #output layer as well

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
    return norm_dataset

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

