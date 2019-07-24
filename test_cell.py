#!/usr/bin/python
# -*- coding: UTF-8 -*-

import copy

# class
class Perceptron:
    def __init__(self, input_num, activator):
        '''
        init the feature dimension & f
        '''
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0
    def predict(self, value_vec):
        '''
        input data & output predient value.
        '''
        # f(value * weight + b)
        #value = value_vec * self.weights + self.bias
        value = 0
        for xx in range(len(value_vec)):
            value += value_vec[xx] * self.weights[xx] + self.bias
        output = self.activator(value)
        return output
    def update_weights(self,value_input, value_true, value_pred, delta):
        diff = value_true - value_pred
        delta_w = [0,0]
        for d in range(len(value_input)):
            delta_w[d] = delta * diff * value_input[d]
        delta_b = delta * diff


        for x in range(len(self.weights)):
            self.weights[x] = self.weights[x] + delta_w[x]
        self.bias = self.bias + delta_b
    def train_data(self,value_vec, value_labels, delta):
        '''
        '''
        lens = len(value_vec)
        for value in range(lens):
            input_unit = value_vec[value]
            input_label_unit = value_labels[value]
            out = self.predict(input_unit)
            self.update_weights(input_unit, input_label_unit, out, delta)
        print('weights',self.weights, 'bias',self.bias)
    def learn(self, input_vec, input_labels, delta, iteators):
        '''
        '''
        for i in range(iteators):
            self.train_data(input_vec, input_labels, delta)




def fun(x):
    if x > 0:
        return 1
    else:
        return 0

def getTrainData():
    input_vec = [[1,1], [0,0], [1,0], [0,1]]
    input_labels = [1,0,0,0]
    return input_vec, input_labels


p = Perceptron(2,fun)
input_data , input_labels= getTrainData()
p.learn(input_data, input_labels, 0.1, 10)


print '1 and 1 = %d' % p.predict([1, 1])
print '0 and 0 = %d' % p.predict([0, 0])
print '1 and 0 = %d' % p.predict([1, 0])
print '0 and 1 = %d' % p.predict([0, 1])