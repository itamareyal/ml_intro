'''
    Machine learning python project 1
    Michal Keren 204783161
    Itamar Eyal 302309539
'''

#   mnist.py- main script of the program

'''
    IMPORTS
'''
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
    set_one_hot = (mask == raw_set).astype('float16')
    # add check for 0 & 1 values, convert to 0.01 & 0.99
    set_one_hot[set_one_hot == 0] = 0.01
    set_one_hot[set_one_hot == 1] = 0.99
    return set_one_hot


class MNIST:

    def __init__(self, X_train, X_test, X_val, t_train, t_test, t_val, W):
        # X sets & one hot form
        self.X_train = X_train
        self.X_train_OH = get_one_hot(X_train)

        self.X_test = X_test
        self.X_test_OH = get_one_hot(X_test)

        self.X_val = X_val
        self.X_val_OH = get_one_hot(X_val)

        # t sets & one hot form
        self.t_train = t_train
        self.t_train_OH = get_one_hot(t_train)

        self.t_test = t_test
        self.t_test_OH = get_one_hot(t_test)

        self.t_val = t_val
        self.t_val_OH = get_one_hot(t_val)

        self.N = X_train.shape[0]
        self.W = W
        self.eta = 0.01


    def get_y(self, train_vector, k):
        W = self.W

        numerator = math.exp(np.transpose(W[k]) * train_vector)
        denominator = 0
        for element in range(NUMBER_OF_LABELS):
            denominator += math.exp(np.transpose(W[element]) * train_vector)
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
        X_OH = self.X_train_OH
        N = self.N
        t = self.t_train

        for n in len(range(X)):
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
#X = mnist['data'].astype('float64')
X = mnist['data'].astype('float16')#each row is a 28x28 photo.
t = mnist['target'].astype('int16')

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
#X = X.reshape((X.shape[0], 785)) #This line flattens the image into a vector of size 784

#3- construct the X matrix.
X= np.c_[X, np.ones(X.shape[0]).astype('float16')] # adding '1' to the end of each photo vector.
#X=[x0^T,x1^T...,x9^T]^T

#4- split the DataBse into: training set- 60%, validation set- 20%, test set-20%.
X_train, X_test, t_train, t_test = train_test_split(X, t, train_size= 0.6)
X_test, X_val,t_test, t_val= train_test_split(X_test, t_test, test_size= 0.5) # split half of the test_set into validation set.

#5 - initialize the Wights vectors
W = np.random.rand(10,785).astype('float16') #W=[w0^T,w1^T...,w9^T]^T

# initialize the class element
subject = MNIST(X_train, X_test, X_val, t_train, t_test, t_val, W)

subject.train()



#6- the Error function:
#stuck here
#the code below dosent work
# a= W.dot(np.transpose(X_train))
# y= np.exp(a)/np.sum(np.exp(a)) #soft max



# The next lines standardize the images
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)