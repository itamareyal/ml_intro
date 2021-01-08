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
from sklearn.utils import check_random_state


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
    exp = np.exp(-x)
    return np.divide(1,(np.add(1,exp)))

def error(y,t):
    return np.divide(np.sum(np.power( np.add(y, - t) , 2 )) , 8)

def build_t_train(x):
    sums = np.sum(x, axis=1)
    t = np.ones_like(sums)
    t[np.mod(sums, 2)==0] = 0
    return t

def build_w():
    return np.random.normal(0, 1, size=(4, 4))

'''
    EXECUTION
'''

t_train = build_t_train(x_train)

# shuffle the DataBase
random_state = check_random_state(1)
permutation = random_state.permutation(x_train.shape[0]) # a random arrangement of the indexes in the range [0,X.shape[0]]
x_train = x_train[permutation] # shuffling the order of the pictures.
t_train = t_train[permutation] # arranging the correct labels accordingly.
t_train = np.reshape(t_train, (8,1))

w= build_w()
print(x_train)
print(w)
print(w[0])


# E = np.zeros(size=(100,2000))

# calculates delta for the output node y
def get_delta_output(t,y):
    h_prime = np.multiply(y, np.add(1,-y))
    delta_out = np.multiply(h_prime, np.add(y,-t))
    return delta_out

def get_delta_hl(delta, z, w_row):
    delta_resized = np.tile(delta,(1,3))
    w_resized = np.tile(w_row.T,(8,1))

    h_prime = np.multiply(z, np.add(1, -z))
    h_w = np.multiply(h_prime, w_resized)
    delta_hl = np.multiply(delta_resized, h_w) # add bias here? is the bias a node?
    return delta_hl

# calculate ai
def get_ai(z, row_w):
    return np.dot(row_w[1:],z.T) + row_w[0] * np.ones(shape=(1,8))


def parity_3(x, t):
    E = np.zeros(shape=(1,2000))
    w = build_w()
    a = np.zeros(shape=(8,3))
    for i in range(3):
        a[:,i]=get_ai(x, w[i])
    print("a: \n",a)
    # get z of the only hidden layer
    z = logistic_sigmoin(a)
    print("z: \n",z)

    a_out = get_ai(z, w[3])
    y = logistic_sigmoin(a_out).T
    print("y: \n", y)

    E = []
    E.append(error(y,t))
    for iter in range(2000):
        delta_out  =get_delta_output(t,y)
        print(delta_out)
        get_delta_hl(delta_out, z, w[3][1:])
        #w = w - ETA * np.sum(delta_out,
    return

parity_3(x_train,t_train)


