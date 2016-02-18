import numpy as np
import matplotlib
import os

def load_dataset():
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print ("Downloading ", filename)
        import urllib
        urllib.urlretrieve(source+filename, filename)

    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)

        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            data = data.reshape(-1,1,28,28)

        return data/np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)

        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)

        return data

    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    Y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    Y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = load_dataset()

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
#plt.show(plt.imshow(X_train[3][0]))

import theano
import lasagne
import theano.tensor as T
def build_NN(input_var=None):

    #First we have an input layer - the expected input shape is 1x28x28 (for 1 image)
    #We will link this input layer to the input_var (which will be the array of images)
    l_in = lasagne.layers.InputLayer(shape=(None,1,28,28), input_var=input_var)

    #Let's add a 20% dropout - this means that randomly 20% of the edges between the input layer and the next layer will be dropped.
    #This is done to avoid overfitting (your model is too highly designed for your training data)
    l_in_drop = lasagne.layers.DropoutLayer(l_in,p=0.2)

    #Add a layer with 800 nodes.  Initially this will bea  dense / fully-connected (i.e. every edge that is possible will exist)
    l_hid1 = lasagne.layers.DenseLayer(l_in_drop, num_units=800, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())

    #This layer has been initialized with some weights.  There are some schemes to initialize the weights so that training will be done faster.

    #We will now add a dropout of 50% to the hidden layer 1
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1,p=0.5)

    #Add another layer the same way
    l_hid2 = lasagne.layers.DenseLayer(l_hid1_drop, num_units=800, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2,p=0.5)

    #the output layer has 10 units. softmax specifies that each of those ouputs is between 0-1 and the max of those will be the final prediction
    l_out = lasagne.layers.DenseLayer(l_hid2_drop, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

#We've set up the network.  Now we have to tell the network how to train itself (i.e. how should it find the values of all the weights it needs to find)

#We'll initialize some empty arrays which will act as placeholders for the training/test data that will be given to the network
input_var = T.tensor4('inputs')  #empty 4 dimensional array
target_var = T.ivector('targers') #an empty 1 dimensional integer array

network = build_NN(input_var)

#In training, we are going to follow the steps below

#a. compute an error function
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction,target_var)
loss = loss.mean()

#b. We'll tell the network how to update all of its weights based on the value of the error function
params = lasagne.layers.get_all_params(network, trainable=True)  #current value of all weights
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

#We'll use theano to compile a function that is going to represent a single training step i.e. compute the error, find the current weights, update the weights
train_fn = theano.function([input_var, target_var], loss, updates=updates)
#calling this function for a certain number of times will train the NN until you reach a minimum error value

num_training_steps=10 #ideally you can train for a few 100 steps

for step in range(num_training_steps):
    train_err = train_fn(X_train, Y_train)
    print("Current step is " + str(step))

#To check the prediction for 1 image, we'll need to set up another function

test_prediction = lasagne.layers.get_output(network)
val_fn = theano.function([input_var], test_prediction)

print(val_fn([X_test[0]]))  #this will apply the function on 1 image, the first one in the test set

#let's check the actual value
print(Y_test[0])

#Last step, we'll feed a test data of 10000 images to the trained neural network and check its accuracy
#We'll set up a function that will take in images and their lables, feed the images to our network and compute its accuracy
#Deterministic = true means that the dropout doesn't happen as training occurs.
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

#Checks the index of the max value in each test prediction and matches it against the actual value
acc_fn = theano.function([input_var, target_var], test_acc)

print("Accuracy : ", acc_fn(X_test,Y_test))
