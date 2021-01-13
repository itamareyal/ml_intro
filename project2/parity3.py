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

'''
    EXECUTION
'''

t_train = build_t_train(x_train)



w= build_w()
# print(x_train)
# print(w)
# print(w[0])


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

    w = build_w()
    a = np.zeros(shape=(8,3))
    E = np.zeros(2000)





    for iter in range(ITERATIONS):

        for i in range(3):
            a[:, i] = get_ai(x, w[i])
        # print("a: \n",a)
        # get z of the only hidden layer
        z = logistic_sigmoid(a)
        # print("z: \n",z)

        a_out = get_ai(z, w[3])
        y = logistic_sigmoid(a_out).T
        # print("y: \n", y)

        E[iter] = error(y, t)


        delta_out  =get_delta_output(t,y)
        #print(delta_out)
        delta_hl = get_delta_hl(delta_out, z, w[3][1:])


        w[3][1:] = np.add(w[3][1:] , - ETA * np.sum(np.multiply(delta_out, z), axis=0))

        for node in range(3):
            w[node][1:] = np.add(w[node][1:] , - ETA * np.sum(np.multiply(delta_hl, x), axis=0))

        # prog = iter * 100 / 2000
        # sys.stdout.write("\r%d%% " % prog)
        # sys.stdout.flush()
    return E


E_mat = np.zeros(shape=(100,2000))
for turn in range(TURNS):
    # print('turn ' + str(turn))
    # shuffle the DataBase
    random_state = check_random_state(1)
    permutation = random_state.permutation(x_train.shape[0]) # a random arrangement of the indexes in the range [0,X.shape[0]]
    x_train = x_train[permutation] # shuffling the order of the pictures.
    t_train = t_train[permutation] # arranging the correct labels accordingly.
    t_train = np.reshape(t_train, (8,1))
    err_vector = parity_3(x_train,t_train)
    # print('err calculated')
    E_mat[turn] = err_vector


results = np.mean(E_mat, axis=0)
print('results: ' +str(results))
plt.figure(0)
plt.plot(results)
plt.suptitle('Part A - Mean square error by iteration:', fontsize=12, fontweight='bold')
plt.xlabel("iteration")
plt.ylabel("MSE")
plt.show()



