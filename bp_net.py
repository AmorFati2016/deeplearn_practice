#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

import sys


class ActivatorSigmoid(object):
    def __init__(self):
        self.input = []
    def forward(self, input_data):
        self.input = []
        # print('enter activator forward')
        return 1.0/(1.0 + np.exp(-input_data))

    def backward(self, input_data):
        # print("enter activator backward")
        return input_data * (1 - input_data)


class FullConnectionLayers(object):
    def __init__(self, input_num, output_num, activator):
        self.input_size = input_num
        self.output_size = output_num
        self.output = np.zeros((output_num, 1))
        self.input  = np.zeros((input_num,1))
        self.bias = np.zeros((output_num, 1))
        self.activator = activator
        self.w = np.random.normal(-0.1, 0.1, (output_num, input_num))

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activator.forward(np.dot(self.w, input_data) + self.bias)

    def backward(self, input_data):
        self.delta = self.activator.backward(self.input) * np.dot(self.w.T, input_data)
        self.w_grad = np.dot(input_data, self.input.T)
        self.b_grad = input_data

    def updateWight(self, delta):
    	self.w = self.w + self.w_grad * delta
    	self.bias = self.bias + self.b_grad * delta


class Network(object):
    def __init__(self, netinfo):
        # print('init the net work')
        self.layers = []
        activator = ActivatorSigmoid()
        for i in range(len(netinfo) - 1):
            layer = FullConnectionLayers(netinfo[i], netinfo[i + 1], activator)
            self.layers.append(layer)

    def train(self, data, labels, delta, itea_num):
        '''
        '''
        for itea in range(itea_num):
            self.train_one(data, labels, delta)
            error = self.loss(self.predict(data[-1]), labels[-1])
            print(" error is ", error)
    def loss(self, label, output):
    	return 0.5 * ((label - output) * (label - output)).sum()

    def train_one(self, data, labels, delta):
        for index in range(len(data)):
            self.predict(data[index])
            self.calc_gradient(labels[index])
            self.update_weight(labels[index], delta)

    def predict(self, data):
        # print("predict")
        output = data
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def calc_gradient(self, label):
         # compute last layer delta
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
        	layer.backward(delta)
        	delta = layer.delta
        # print('calc_gradient')

    def update_weight(self, labels, delta):
        '''
        '''
        for layer in self.layers:
        	layer.updateWight(delta)
        # print('update weight')


class Normalizer(object):
    def __init__(self):
        self.mask = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]

    def norm(self, number):
        data = map(lambda m: 0.9 if number & m else 0.1, self.mask)
        return np.array(data).reshape(8, 1)


# train data
# step 1. get train data
def get_train_data():
    ''''''
    print('get train data')
    init_data = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 255, 8):
        input_vec = init_data.norm(i)
        label = input_vec
        data_set.append(input_vec)
        labels.append(label)
    return data_set, labels


# begin train
def train_data():
    '''
    '''
    # step 1 get train data
    data, labels = get_train_data()
    # step 2 init network
    net = Network([8, 7, 8])
    # step 3.0 train
    delta = 0.5
    num = 100
    net.train(data, labels, delta, num)


train_data()
