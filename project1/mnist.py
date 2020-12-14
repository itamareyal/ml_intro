'''
    Machine learning python project 1
    Michal Keren 204783161
    Itamar Eyal 302309539
'''

#   mnist.py- main script of the program

'''
    IMPORTS
'''
from datetime import datetime
start_time = datetime.now()

import matplotlib.pyplot as plt
import math
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import numpy as np


'''
    DEFINES
'''
NUMBER_OF_LABELS = 10
IMG_SIZE = 28
VECTOR_SIZE = 785


'''
    IMPLEMENTATIONS
'''

def get_one_hot(raw_set):
    # Return the vector with the real label marked 1, others marked 0
    mask = np.arange(NUMBER_OF_LABELS)
    set_one_hot = (mask == raw_set).astype('float64')
    # add check for 0 & 1 values, convert to 0.01 & 0.99
    set_one_hot[set_one_hot == 0] = 0.01
    set_one_hot[set_one_hot == 1] = 0.99
    return set_one_hot


class MNIST:

    def __init__(self, X_train, X_test, X_val, t_train, t_test, t_val, W):
        # X sets & one hot form
        self.X_train = X_train
        #self.X_train_OH = get_one_hot(X_train)

        self.X_test = X_test
        #self.X_test_OH = get_one_hot(X_test)

        self.X_val = X_val
        #self.X_val_OH = get_one_hot(X_val)

        # t sets & one hot form
        self.t_train = t_train
        #self.t_train_OH = get_one_hot(t_train)

        self.t_test = t_test
        #self.t_test_OH = get_one_hot(t_test)

        self.t_val = t_val
        #self.t_val_OH = get_one_hot(t_val)

        self.N = X_train.shape[0]
        self.W = W
        self.eta = 0.01


    def get_y(self, train_vector, k):
        W = self.W

        numerator = math.exp(np.dot(W[k] , np.transpose(train_vector)))
        denominator = 0
        for element in range(NUMBER_OF_LABELS):
            denominator += math.exp(np.dot(np.transpose(W[element]) , train_vector))
        return numerator / denominator


    def loss(self, y, n):
        loss =0
        t = self.t_train

        for k in range(NUMBER_OF_LABELS):
            loss += np.dot(t[n][k], math.log(y, math.e))

        return -loss
        # loss =0
        # W = self.W
        # X = self.X_train
        # N = self.N
        # t = self.t_train
        #
        # for n in range(1, N + 1): # 1 to N
        #     row =0
        #     for k in range(NUMBER_OF_LABELS): # 0 to 9
        #         row += t[n][k] * math.log(MNIST.get_y(self,n,k), math.e)
        #     loss += row
        #
        # return -loss


    def get_grad(self, j, y):

        return np.dot(np.transpose(X), (y-t[j]))
        # X = self.X_train
        # t = self.t_train

        # grad =0
        # X = self.X_train
        # N = self.N
        # t = self.t_train
        #
        # for n in range(1, N + 1): # 1 to N
        #     # sigma 1 to N: (y-t)x
        #     grad += (MNIST.get_y(self,n,j) - t[n][j]) * X[n]
        # return grad





    def train(self):

        W = self.W
        X = self.X_train
        #X_OH = self.X_train_OH
        N = self.N
        t = self.t_train

        for n in range(50000):
            # Every iteration trains by 1 img and corrects the W matrix
            y_vector = np.array((1, NUMBER_OF_LABELS))

            # generate y
            for k in range(NUMBER_OF_LABELS):
                y_vector[k] = MNIST.get_y(self, X_train[n], k)

            y = np.argmax(y_vector)

            # correct W
            for j in range(NUMBER_OF_LABELS):
                grad = MNIST.get_grad(self, j, y)
                W[j] -= self.eta * grad

            # get loss of iteration by train_set
            loss = MNIST.loss(self, y_vector, n)

            # get accuracy of iteration by Val_set



'''
    EXECUTION
'''


#1- load MNIST DataBase.
mnist = fetch_openml('mnist_784')
X = mnist['data'].astype('float64')#each row is a 28x28 photo.
t = mnist['target'].astype('int64')

#transform t into hot vectors array:
h = np.zeros((t.size, 10))
h[np.arange(t.size),t] = 1
t=h

# shuffle the DataBase
random_state = check_random_state(1)
permutation = random_state.permutation(X.shape[0]) # a random arrangement of the indexes in the range [0,X.shape[0]]
X = X[permutation] # shuffling the order of the pictures.
t = t[permutation] #aranging the correct labels accordingly.

#2- flatten the data from pictures to vectors.
#its already done in the begging, isn't it?
X = X.reshape((X.shape[0], -1)) #This line flattens the image into a vector of size 784

#3- construct the X matrix.
X= np.c_[X, np.ones(X.shape[0]).astype('float64')] # adding '1' to the end of each photo vector.
#X=[x0^T,x1^T...,x9^T]^T

#4- split the DataBse into: training set- 60%, validation set- 20%, test set-20%.
X_train, X_test, t_train, t_test = train_test_split(X, t, train_size= 0.6)
X_test, X_val,t_test, t_val= train_test_split(X_test, t_test, test_size= 0.5) # split half of the test_set into validation set.

#5 - initialize the Wights vectors, values [0,1]
W = np.random.rand(10,785).astype('float64') #W=[w0^T,w1^T...,w9^T]^T
W[W == 0] = 0.01

# The next lines standardize the images
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)
tests =0
prev =0
prev_correct_p =0

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return np.divide(exp_x , np.sum(exp_x, axis=1, keepdims=True))


f = open("mnist.txt", "a")
f.write("\n\nNew test called at: "+str(start_time)+"\n")


while tests < 50:
#6- the Error function:
    #fac = 0.99 / 255
    #X_train = np.multiply(X_train , fac)
    a= W.dot(X_train.T).T
    #f.write("\n\ntrain "+str(tests)+":\n")
    #f.write("a "+str(a)+"\n")
    #print(a)
    y = softmax(a)

    #f.write("y "+str(y)+"\n")
    #exps = np.exp(a-np.max(a))
    #y = np.divide(exps , np.sum(exps))
    #y= np.divide(np.exp(a),np.sum(np.exp(a))) #soft max
    #print(y)

    #log_likelihood = -np.log(y[range(N), t_train])
    #bc = t_train.dot(np.log(y.T))
    cc = t_train.T.dot(np.log(y))
    #f.write("cc "+str(cc)+"\n")
    cel= -np.sum(cc)
    #cel = np.sum(log_likelihood) / N
    print ("test "+str(tests)+": error="+str(cel))
    #f.write("test "+str(tests)+": error="+str(cel)+"\n")
    if prev != 0:
        print("improved by: "+ str(prev - cel)+"\n")

    prev = cel
    grad = X_train.T.dot(y-t_train)
    eta = 0.009
    #grad_no_bias = grad[:-1, :]
    #grad_no_bias = np.c_[grad_no_bias, np.zeros(grad_no_bias.shape[0]).astype('float64')]
    # grad_no_bias = grad
    # grad_no_bias[:, 9] = 0
    W = W - np.multiply(eta, grad.T)


    # Validation
    #X_val = np.multiply(X_val , fac)
    b = np.transpose(W.dot(np.transpose(X_val)))
    yv = softmax(b) #soft max
    guesses = yv.argmax(axis = 1)
    answers = t_val.argmax(axis=1)
    corrects_mat = np.equal(guesses, answers)
    corrects = np.sum(corrects_mat)
    print("correct guesses: "+str(corrects))
    correct_p = corrects/guesses.shape[0] * 100
    print("correct in t_val: "+str(corrects/guesses.shape[0] * 100)+"%")

    print("%.2f" % correct_p + "%")
    f.write("correct guesses: "+str(corrects)+"\n")
    f.write("correct in t_val: "+"%.2f" % correct_p+"%\n")

    if abs(prev_correct_p - correct_p) < 0.2 and correct_p > 90:
        break
    else:
        prev_correct_p = correct_p

    
    # increment test
    tests += 1
f.write("Total running time: "+str(datetime.now() - start_time)+"\n")
f.write("Total iterations: "+str(tests)+"\n")
f.write("Program finished successfully.\n")
f.close()