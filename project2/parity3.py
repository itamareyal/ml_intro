'''
    Machine learning python project 2
    Michal Keren 204783161
    Itamar Eyal 302309539

Libraries: numpy, matplotlib.
'''

'''
    IMPORTS
'''
import matplotlib.pyplot as plt
import numpy as np


'''
    DEFINES
'''
x_train = np.array([[0,0,0],
                    [0,0,1],
                    [0,1,0],
                    [0,1,1],
                    [1,0,0],
                    [1,0,1],
                    [1,1,0],
                    [1,1,1]])

ETA = 2


'''
    IMPLEMENTATIONS
'''

def logistic_sigmoin(x):
    return np.divide(1,(1+np.exp(-x)))

def error(y,t):
    return np.divide(np.power( (y - t) , 2 ) , 8)

def build_t_train(x):
    sums = np.sum(x, axis=1)
    t = np.ones_like(sums)
    t[np.mod(sums, 2)==0] = 0
    return t

def build_w():
    return np.random.normal(0, 1, size=(2, 4))

print(build_t_train(x_train))
print(build_w())