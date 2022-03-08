#Author: Monynich Kiem
#Date: 03/08/2022
#Desc: CS460G Neural Network + Multilayer Perceptron

import numpy as np
import pandas as pd
import math

def sigmoid(exponent):
    sigmoid = 1 / (1 + np.exp(-exponent))
    return sigmoid

print(sigmoid(2)) #it works great moving on


