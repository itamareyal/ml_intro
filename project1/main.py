'''
    Machine learning python project 1
    Michal Keren 204783161
    Itamar Eyal 302309539
'''

#   main.py- main script of the program

'''
    IMPORTS
'''
import numpy as np
import matplotlib.pyplot as plt
import math


'''
    DEFINES
'''
NUMBER_OF_LABELS = 10
IMG_SIZE = 28


'''
    IMPLEMENTATIONS
'''

def img_to_vector(img):
    # Flattens the img into a 1D vector and adds 1 at the end
    d1, d2 = img.shape
    img_enrolled = img.reshape((d1 * d2, 1))
    return np.append(img_enrolled, 1)


def get_one_hot(raw_set):
    # Return the vector with the real label marked 1, others marked 0
    mask = np.arange(NUMBER_OF_LABELS)
    set_one_hot = (mask == raw_set).astype(np.float)
    # add check for 0 & 1 values, convert to 0.01 & 0.99
    return set_one_hot


def create_w_vector():
    w = np.random.rand(IMG_SIZE * IMG_SIZE + 1)
    return np.append(w, 1)


def create_w_matrix():
    w = np.empty(shape=(NUMBER_OF_LABELS, IMG_SIZE * IMG_SIZE + 1))
    for row in range(NUMBER_OF_LABELS):
        np.append(w , create_w_vector())
    return w


class MNIST:

    def __init__(self, test_set, training_set, validation_set):
        self.TestSet = test_set
        self.TestSetOneHot = get_one_hot(test_set)

        self.TrainingSet = training_set
        self.TrainingSetOneHot = get_one_hot(training_set)
        self.N = training_set.shape[0]

        self.ValidationSet = validation_set
        self.ValidationSetOneHot = get_one_hot(validation_set)

        self.w = create_w_matrix()
        self.eta = 0.01


    def create_x_matrix(self):
        x = np.empty(shape=(self.N, IMG_SIZE * IMG_SIZE + 1))
        for img in self.TrainingSet:
            np.append(x, img_to_vector(img))
        return x


    def get_y(self, n, k, w, x):
        numerator = math.exp(np.transpose(w[k]) * x[n])

        denominator = 0
        for element in range(NUMBER_OF_LABELS):
            denominator += math.exp(np.transpose(w[element]) * x[n])

        return numerator / denominator


    def get_t(self, n, k):
        # To be implemented
        return 1


    def loss(self, x):
        loss =0
        w = self.w

        for n in range(1, self.N + 1): # 1 to N
            row =0
            for k in range(NUMBER_OF_LABELS): # 0 to 9
                row += MNIST.get_t(self,n,k) * math.log(MNIST.get_y(self,n,k,w,x), math.e)
            loss += row

        return -loss


    def get_grad(self, x, j):
        grad =0
        for n in range(1, self.N + 1): # 1 to N
            # sigma 1 to N: (y-t)x
            grad += (MNIST.get_y(self,n,j,self.w,x) - MNIST.get_t(self,n,j)) * x[n]
        return grad
