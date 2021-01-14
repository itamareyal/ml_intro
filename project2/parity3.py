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
import sys


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
ITERATIONS = 2000
TURNS = 100

'''
    IMPLEMENTATIONS
'''

def logistic_sigmoid(x):
    exp = np.exp(-x)
    return np.divide(1,(np.add(1,exp)))

def error(y,t):
    return np.sum(np.power( np.add(y, - t) , 2 )) / 8

def build_t_train(x):
    sums = np.sum(x, axis=1)
    t = np.ones_like(sums)
    t[np.mod(sums, 2)==0] = 0
    return t

def build_w():
    return np.random.normal(0, 1, size=(4, 4))

def build_wB():
    return np.random.normal(0, 1, size=(10, 4))

def get_delta_output(t, y):
    h_prime = np.multiply(y, np.add(1, -y))
    delta_out = np.multiply(h_prime, np.add(y, -t))
    return delta_out


def get_delta_hl(delta, z, w_row):
    delta_resized = np.tile(delta, (1, 3))
    w_resized = np.tile(w_row.T, (8, 1))

    h_prime = np.multiply(z, np.add(1, -z))
    h_w = np.multiply(h_prime, w_resized)
    delta_hl = np.multiply(delta_resized, h_w)
    return delta_hl

def get_delta_hlB(delta_vec, z, w_prev_layer):
    h_prime = np.multiply(z, np.add(1, -z))
    sum_over_k = np.zeros(shape=(8,3))
    for row in range(3):
        w_row = w_prev_layer[row][1:]
        w_resized = np.tile(w_row.T, (8, 1))
        w_delta = np.multiply(w_resized, delta_vec)
        sum_over_k = np.add(sum_over_k,w_delta)

    delta_hl = np.multiply(h_prime, sum_over_k)
    return delta_hl


def get_delta_hl_dispath(delta_vec, z, w_prev_layer):
    h_prime = np.multiply(z, np.add(1, -z))
    sum_over_k = np.zeros(shape=(8,3))
    for row in range(3): # row of w for one node
        w_row = w_prev_layer[row][1:]
        for i in range(3): # selecting delta
            delta = delta_vec[:,i]
            delta_hl = get_delta_hl(delta.reshape(8,1), z, w_row)
            sum_over_k = np.add(sum_over_k,delta_hl)

    delta_hl = np.multiply(h_prime, sum_over_k)
    return delta_hl

def get_ai(z, row_w):
    return np.dot(row_w[1:], z.T) + row_w[0] * np.ones(shape=(1, 8))

def parity_3(x, t):
    w = build_w()  # new randomized weights for the turn
    a = np.zeros(shape=(8, 3))  # holds x*w
    E = np.zeros(2000)  # holds MSE by iteration for the turn

    for iter in range(ITERATIONS):
        # fill a mat for every node of the hidden layer
        for i in range(3):
            a[:, i] = get_ai(x, w[i])
        z = logistic_sigmoid(a)

        # fill a mat for output node
        a_out = get_ai(z, w[3])
        y = logistic_sigmoid(a_out).T

        # calculate MSE
        E[iter] = error(y, t)

        delta_out = get_delta_output(t, y)
        delta_hl = get_delta_hl(delta_out, z, w[3][1:])

        # update weights for output node
        w[3][1:] = np.add(w[3][1:], - ETA * np.sum(np.multiply(delta_out, z), axis=0))

        # update weights for hidden layer nodes
        for node in range(3):
            w[node][1:] = np.add(w[node][1:], - ETA * np.sum(np.multiply(delta_hl, x), axis=0))

    return E


def parity_3B(x, t):
    w = build_wB()  # new randomized weights for the turn
    a = np.zeros(shape=(8, 3))  # holds x*w for a specific, one hidden layer
    E = np.zeros(2000)  # holds MSE by iteration for the turn
    w1 = w
    for iter in range(ITERATIONS):
        # fill a mat for every node of the hidden layer
        layer = 0
        for i in range(3):
            a[:, i] = get_ai(x, w[layer])
            layer +=1
        z1 = logistic_sigmoid(a)

        for i in range(3):
            a[:, i] = get_ai(z1, w[layer])
            layer += 1
        z2 = logistic_sigmoid(a)

        for i in range(3):
            a[:, i] = get_ai(z2, w[layer])
            layer += 1
        z3 = logistic_sigmoid(a)

        # fill a mat for output node
        a_out = get_ai(z3, w[layer])
        y = logistic_sigmoid(a_out).T

        # calculate MSE
        E[iter] = error(y, t)

        delta_vector = np.zeros(shape=(8,10))
        delta_out = get_delta_output(t, y)
        delta_vector[:,9] = delta_out.reshape(8)
        z_layers = [x,z1,z2,z3]

        w[9][1:] = np.add(w[9][1:], - ETA * np.sum(np.multiply(delta_out, z3), axis=0))
        for node in range(8,-1,-1):
            layer = int(node/3)
            delta_hl = get_delta_hl(delta_vector[:,node].reshape(8,1), z_layers[layer], w[node][1:])
            delta_vector[:,node-1] = np.sum(delta_hl,axis=1)

            w[node][1:] = np.add(w[node][1:], - ETA * np.sum(np.multiply(delta_hl, z_layers[layer]), axis=0))


        # for layer in range(2,-1,-1):
        #     for node in range(2,-1,-1):
        #         n = layer*3 + node
        #         delta_vector[:, n] = get_delta_hl(delta_vector[:,n].reshape(8,1), z_layers[layer], w[n][1:])
        # delta_hl3 = get_delta_hl(delta_out, z3, w[9][1:])
        # delta_hl2 = get_delta_hl_dispath(delta_hl3, z2, w[6:9])
        # delta_hl1 = get_delta_hl_dispath(delta_hl2, z1, w[3:6])

        # deltas = np.zeros(10)
        # deltas[9] = delta_out
        # z_arr = np.ndarray([z1,z2,z3])
        # for node in range(8,-1,-1): #back propagation on every node
        #
        #     if node in [8,7,6]:
        #         layer = 2
        #     elif node in [5,4,3]:
        #         layer = 1
        #     else:
        #         layer = 0
        #     delta_hl3 = get_delta_hl(deltas[node+1], z_arr[layer], w[3][1:])
        #     delta_hl2 = get_delta_hl(delta_out, z2, w[2][1:])
        #     delta_hl1 = get_delta_hl(delta_out, z1, w[1][1:])

        # update weights for output node
        # w[9][1:] = np.add(w[9][1:], - ETA * np.sum(np.multiply(delta_out, z3), axis=0))
        #
        # # update weights for hidden layer nodes
        # w[6:9] = update_wB(w[6:9], delta_hl3, z2)
        # w[3:6] = update_wB(w[3:6], delta_hl2, z1)
        # w[0:3] = update_wB(w[0:3], delta_hl1, x)

    return E

def update_wB(w_layer, delta, z):
    for node in range(2,-1,-1):
        w_layer[node, 1:] = np.add(w_layer[node, 1:], - ETA * np.sum(np.multiply(delta, z), axis=0))
    return w_layer

'''
    EXECUTION - PART A
'''

t_train = build_t_train(x_train)

E_mat = np.zeros(shape=(100,2000))

for turn in range(TURNS):

    # shuffle the DataBase, every turn has a new ordered data set
    random_state = check_random_state(1)
    permutation = random_state.permutation(x_train.shape[0]) # a random arrangement of the indexes in the range [0,X.shape[0]]
    x_train = x_train[permutation] # shuffling the order of the pictures
    t_train = t_train[permutation] # arranging the correct labels accordingly

    t_train = np.reshape(t_train, (8,1))

    # calls parity3 to get MSE vector
    err_vector = parity_3(x_train,t_train)

    E_mat[turn] = err_vector

# mean of MSE of all turns by iteration
results = np.mean(E_mat, axis=0)


# plot
plt.figure(0)
plt.plot(results)
plt.suptitle('Part A - Mean square error by iteration:', fontsize=12, fontweight='bold')
plt.xlabel("iteration")
plt.ylabel("MSE")
plt.show()


'''
    EXECUTION - PART B
'''
