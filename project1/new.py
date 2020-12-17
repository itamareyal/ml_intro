'''
    Machine learning python project 1
    Michal Keren 204783161
    Itamar Eyal 302309539
Libraries: numpy, matplotlib & sklearn.
After running the file, 2 plots will show for sections 8,9.
Results for section 10 will be printed to mnist_302309539_204783161.txt at the same folder of this script.
Maximal number of iteration is 50. normally is finished after about 10-14.
Running time of about 35 sec.
'''


'''
    IMPORTS
'''
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import numpy as np


'''
    DEFINES
'''
NUMBER_OF_LABELS = 10  # change to classes
IMG_SIZE = 28


'''
    IMPLEMENTATIONS
'''
def hot_vectors(t):
    h = np.zeros((t.size, 10))
    h[np.arange(t.size), t] = 1
    return h

def softmax(W,X):
    a = W.dot(X.T).T.astype('float64')
    exp_a = np.exp(a - np.max(a, axis=1, keepdims=True))
    return np.divide(exp_a , np.sum(exp_a, axis=1, keepdims=True))

def cross_entropy_loss(y, t):
    #cc = t_train.T.dot(np.log(y))
    # loss =0
    # for n in range(1, t.shape[0]):
    #     for k in range(10):
    #
    #         if t[n][k] == 1 and y[n][k]>0:
    #             loss += -np.log(y[n][k])
    # a = W.dot(X.T).T.astype('float64')
    # exp_a = np.exp(a - np.max(a, axis=1, keepdims=True))
    # sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)
    # rel_exp_a= exp_a[t==1]
    # #rel_sum_exp_a = sum_exp_a[t==1]
    # lny1= np.log(rel_exp_a)
    # lny2=-np.log(sum_exp_a)
    # E=-np.sum(lny1+lny2)
    E= -np.sum(np.log(y[t==1]))
    print(E)
    return E

# def precision_on_set(W, X_set, T_set, set_name, f):
#     #wx = np.transpose(W.dot(np.transpose(X_set)))
#     y_set = softmax(W,X_set) #soft max
#     guess = y_set.argmax(axis = 1)
#     ans = T_set.argmax(axis=1)
#     c_mat = np.equal(guess, ans)
#     c = np.sum(c_mat)
#     f.write("\nFinal results by set for: " + set_name+"\n")
#
#     c_p = c/guess.shape[0] * 100
#     f.write("correct guesses: "+str(c)+"\n")
#     f.write("precision "+set_name+": "+ "%.2f" % c_p+"%\n")

def get_set_accuracy(W,X,t):
    y = softmax(W, X)
    guesses = y.argmax(axis=1)
    answers = t.argmax(axis=1)
    corrects_mat = np.equal(guesses, answers)
    corrects = np.sum(corrects_mat)  # num of correct guesses
    accuracy = corrects / guesses.shape[0] * 100  # correct guesses in %
    return accuracy



'''
    EXECUTION
'''
# 1 & 2- load MNIST DataBase & flatten the data from pictures to vectors.
mnist = fetch_openml('mnist_784')
X = mnist['data'].astype('float16')#each row is a 28x28 photo. (flatten the data from pictures to vectors.)
t = mnist['target'].astype('int64')

t= hot_vectors(t) #transform t into hot vectors array.

# shuffle the DataBase
random_state = check_random_state(1)
permutation = random_state.permutation(X.shape[0]) # a random arrangement of the indexes in the range [0,X.shape[0]]
X = X[permutation] # shuffling the order of the pictures.
t = t[permutation] # arranging the correct labelflatten the data from pictures to vectorss accordingly.


# 3- construct the X matrix.
X= np.c_[X, np.ones(X.shape[0]).astype('float16')] # adding '1' to the end of each photo vector.
                                                   #X=[x0^T,x1^T...,x9^T]^T (10 rows)

# 4- split the DataBse into: training set- 60%, validation set- 20%, test set-20%.
X_train, X_test, t_train, t_test = train_test_split(X, t, train_size= 0.6)
X_test, X_val,t_test, t_val= train_test_split(X_test, t_test, test_size= 0.5) # split half of the test_set into validation set.

# 5 - initialize the Wights vectors, values [0,1]
#W = np.random.rand(NUMBER_OF_LABELS, IMG_SIZE * IMG_SIZE + 1).astype('float16') #W=[w0^T,w1^T...,w9^T]^T (10 rows)
W = np.random.uniform(low=0, high=0.01, size=(NUMBER_OF_LABELS,IMG_SIZE*IMG_SIZE + 1)).astype('float64')


# The next lines standardize the images
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# # open file for output log
# f = open("mnist_302309539_204783161.txt", "a")

# lists to hold plotting data
E_lst = []
VS_accuracy_lst = []

eta = 0.01
precision = 0.0001 #The condition for convergence
max_iters = 10**5 # maximum number of iterations
i =0  # index of iteration
accuracy_diff= precision+1 #init

# The GD iteration loop. every iteration trains the W matrix and calculates the validationSet accuracy.
while accuracy_diff > precision and i < max_iters:
    y = softmax(W,X_train)

    # 6- the Error function:
    E_lst.append(cross_entropy_loss(y, t_train)) # add to list for plotting

    grad_E = X_train.T.dot(y-t_train) # the gradient of loss.
    # 7- gradient descent
    W = W - eta*grad_E.T # update W

    # calc Validation Set accuracy
    valSet_accuracy =get_set_accuracy(W,X_val,t_val)
    if(len(VS_accuracy_lst)>0):
        accuracy_diff = abs(VS_accuracy_lst[-1]-valSet_accuracy)
    VS_accuracy_lst.append(valSet_accuracy)
    i += 1
i_lst = np.arange(0, i)

# 8- Cross Entropy Loss on train set as a function of iteration
plt.scatter(i_lst, E_lst)
plt.plot(i_lst, E_lst)
plt.suptitle('Training Set Loss Vs. iteration', fontsize=14, fontweight='bold')
plt.xlabel("iteration")
plt.ylabel("Cross Entropy Loss")

# 9- Precision as a function of iteration for validation set
plot_cp= plt.figure(2)
plt.scatter(i_lst, VS_accuracy_lst)
plt.plot(i_lst, VS_accuracy_lst)
plt.suptitle('validation set accuracy Vs. iteration', fontsize=14, fontweight='bold')
plt.xlabel("iteration")
plt.ylabel("accuracy[%]")


# 10- results on every set after last iteration
trainSet_accuracy =str("%.2f" % get_set_accuracy(W,X_train,t_train))
testSet_accuracy =str("%.2f" % get_set_accuracy(W,X_test,t_test))

print("------final accuracy Values:------")
print("Training Set accuracy: "+trainSet_accuracy+"%")
print("Test Set accuracy: "+testSet_accuracy+"%")
print("Validation Set accuracy: "+str("%.2f" % VS_accuracy_lst[-1])+"%")

# present plots
plt.show()

# precision_on_set(W,X_train,t_train,"train",f)
# precision_on_set(W,X_val,t_val,"validation",f)
# precision_on_set(W,X_test,t_test,"test",f)
# f.write("Total iterations: "+str(i)+"\n")
# f.write("Program finished successfully.\n")
# f.close()

