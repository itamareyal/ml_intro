from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import numpy as np
#1- load MNIST DataBase.
mnist = fetch_openml('mnist_784')
#X = mnist['data'].astype('float64')
X = mnist['data'].astype('float16')#each row is a 28x28 photo.
t = mnist['target'].astype('int16')

#transform t into hot vectors array:
h = np.zeros((t.size, 10))
h[np.arange(t.size),t] = 1
t=h

# suffel the DataBase
random_state = check_random_state(1)
permutation = random_state.permutation(X.shape[0]) # a random arrangment of the indexes in the range [0,X.shape[0]]
X = X[permutation] # suffeling the order of the pictures.
t = t[permutation] #aranging the correct labels accurdenly.

#2- flatten the data from pictures to vectors.
#its already done in the begging, isnt it?
#X = X.reshape((X.shape[0], 785)) #This line flattens the image into a vector of size 784

#3- construct the X matrix.
X= np.c_[X, np.ones(X.shape[0]).astype('float16')] # adding '1' to the end of each photo vector.
#X=[x0^T,x1^T...,x9^T]^T

#4- split the DataBse into: training set- 60%, validation set- 20%, test set-20%.
X_train, X_test, t_train, t_test = train_test_split(X, t, train_size= 0.6)
X_test, X_val,t_test, t_val= train_test_split(X_test, t_test, test_size= 0.5) # split half of the test_set into validation set.

#5 - initialize the Wights vectors
W = np.random.rand(10,785).astype('float16') #W=[w0^T,w1^T...,w9^T]^T

#6- the Error function:
#stuck here
#the code below dosent work
# a= W.dot(np.transpose(X_train))
# y= np.exp(a)/np.sum(np.exp(a)) #soft max



# The next lines standardize the images
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
