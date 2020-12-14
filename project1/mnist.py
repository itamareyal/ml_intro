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
NUMBER_OF_LABELS = 10
IMG_SIZE = 28


'''
    IMPLEMENTATIONS
'''
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return np.divide(exp_x , np.sum(exp_x, axis=1, keepdims=True))

def cross_entropy(y, t):
    #cc = t_train.T.dot(np.log(y))
    loss =0
    for n in range(1, t.shape[0]):
        for k in range(10):

            if t[n][k] == 1 and y[n][k]>0:
                loss += -np.log(y[n][k])

    return loss

def precision_on_set(W, X_set, T_set, set_name, f):
    wx = np.transpose(W.dot(np.transpose(X_set)))
    y_set = softmax(wx) #soft max
    guess = y_set.argmax(axis = 1)
    ans = T_set.argmax(axis=1)
    c_mat = np.equal(guess, ans)
    c = np.sum(c_mat)
    f.write("\nFinal results by set for: " + set_name+"\n")

    c_p = c/guess.shape[0] * 100
    f.write("correct guesses: "+str(c)+"\n")
    f.write("precision "+set_name+": "+ "%.2f" % c_p+"%\n")


'''
    EXECUTION
'''
# 1- load MNIST DataBase.
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
t = t[permutation] # arranging the correct labels accordingly.

# 2- flatten the data from pictures to vectors.
X = X.reshape((X.shape[0], -1)) #This line flattens the image into a vector of size 784

# 3- construct the X matrix.
X= np.c_[X, np.ones(X.shape[0]).astype('float64')] # adding '1' to the end of each photo vector.
#X=[x0^T,x1^T...,x9^T]^T

# 4- split the DataBse into: training set- 60%, validation set- 20%, test set-20%.
X_train, X_test, t_train, t_test = train_test_split(X, t, train_size= 0.6)
X_test, X_val,t_test, t_val= train_test_split(X_test, t_test, test_size= 0.5) # split half of the test_set into validation set.

# 5 - initialize the Wights vectors, values [0,1]
W = np.random.rand(NUMBER_OF_LABELS, IMG_SIZE * IMG_SIZE + 1).astype('float64') #W=[w0^T,w1^T...,w9^T]^T
W[W == 0] = 0.01 # eliminate zero w elements (at this point they are random)

# The next lines standardize the images
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

tests =0 # index of iterations
prev_correct_p =0 # stop condition

# open file for output log
f = open("mnist_302309539_204783161.txt", "a")

# lists to hold plotting data
cel_lst = []
cp_lst = []
ci_lst = []

# iteration loop. every iteration trains the W matrix and validates it
while tests < 50:
    a= W.dot(X_train.T).T # expression for the exp of y
    y = softmax(a)

    # 6- the Error function:
    cel = cross_entropy(y, t_train) # get CEL
    cel_lst.append(cel) # add to list for plotting

    # 7- gradient descent
    grad = X_train.T.dot(y-t_train) # get gradient E
    eta = 0.015
    W = W - np.multiply(eta, grad.T) # update W

    # Validation
    b = np.transpose(W.dot(np.transpose(X_val))) # expression for exp of y at val set
    yv = softmax(b) # y of val set
    guesses = yv.argmax(axis = 1)
    answers = t_val.argmax(axis=1)
    corrects_mat = np.equal(guesses, answers)
    corrects = np.sum(corrects_mat) # int of correct guesses on val set
    correct_p = corrects/guesses.shape[0] * 100 # correct guesses in %

    # write to log
    f.write("correct guesses: "+str(corrects)+"\n")
    f.write("correct in t_val: "+"%.2f" % correct_p+"%\n")

    # write for lists to plot after loop
    ci_lst.append(tests)
    cp_lst.append(correct_p)

    # check break condition
    if abs(prev_correct_p - correct_p) < 0.2 and correct_p > 90:
        break
    else:
        prev_correct_p = correct_p
    # increment test
    tests += 1


# 8- Cross Entropy Loss on train set as a function of iteration
plt.scatter(ci_lst, cel_lst)
plt.plot(ci_lst, cel_lst)
plt.suptitle('Cross Entropy loss by iteration:', fontsize=14, fontweight='bold')
plt.xlabel("iteration")
plt.ylabel("CEL")

# 9- Precision as a function of iteration for validation set
plot_cp= plt.figure(2)
plt.scatter(ci_lst, cp_lst)
plt.plot(ci_lst, cp_lst)
plt.suptitle('Precision by iteration:', fontsize=14, fontweight='bold')
plt.xlabel("iteration")
plt.ylabel("Precision [%]")

# 10- results on every set after last iteration
precision_on_set(W,X_train,t_train,"train",f)
precision_on_set(W,X_val,t_val,"validation",f)
precision_on_set(W,X_test,t_test,"test",f)
f.write("Total iterations: "+str(tests)+"\n")
f.write("Program finished successfully.\n")
f.close()

# present plots
plt.show()