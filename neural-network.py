# -*- coding: utf-8 -*-
"""
Created on Mon May 17 12:40:07 2021

@author: ariel
"""

import numpy as np

def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def derivative_sigmoid(x):
    return x * (1 - x)

'''
mu, sigma = 0, 0.1
weights = np.array([[0.08279746],
                    [0.02300947],
                    [0.07620112]])

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [0],
              [1],
              [1]])

ephocs = 10000

for i in range(ephocs):
    H_PRED = np.dot(X, weights)
    Y_PRED = sigmoid(H_PRED)
    ERROR = y - Y_PRED

    delta = ERROR * derivative_sigmoid(Y_PRED)

    weights += np.dot(X.T, delta)

print(Y_PRED)

'''
mu, sigma = 0, 0.1
weights1 = np.array([[0.08279746],
                    [0.02300947],
                    [0.07620112]])

syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

X = np.array([[0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

ephocs = 10000

for i in range(ephocs):
    # foward propagation
    l0 = X
    l1 = sigmoid(np.dot(X, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    l2_error = y -l2
    l2_delta = l2_error * derivative_sigmoid(l2)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * derivative_sigmoid(l1)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print(l2)

