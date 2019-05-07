################################### Imports ####################################
from keras.datasets import mnist #from Keras download the mnist dataset
from keras.models import Sequential # import the sequential model
from keras.layers import Dense # import he Dense(Fully connected) layer
from keras.optimizers import rmsprop # import rmsprop optimizer
from keras.utils import to_categorical # prepare the labels
import matplotlib.pyplot as plt #import plot as plt
import os # import os to remove the unwanted msgs from the console
################################################################################

# to remove the unwanted msgs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#load the training and testing data from the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# to view the datset
plt.imshow(train_images[4], cmap= plt.cm.binary)
plt.show()

# Neural Network

model = Sequential() # make our model Sequential
model.add(Dense(50, activation = "relu", input_shape= (28*28,))) # add a hidden layer
model.add(Dense(10, activation = 'softmax')) # add an output layer

# do the back propagation 
# rmsprop optimizer
# model.compile(optimizer = "rmsprop", loss = 'categorical_crossentropy', metrics=['accuracy'])

# Adam optimizer
model.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics=['accuracy'])

# prepare the train data
train_images = train_images.reshape((60000, 28*28))
# Normalize the data
train_images = train_images.astype("float32") / 255
# prepare the test data
test_images = test_images.reshape((10000, 28*28))
# Normalize the test data
test_images = test_images.astype("float32") / 255

# prepare the labels

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Train the Neural network
model.fit(train_images, train_labels, epochs = 5, batch_size = 128)

# Test the Neural network
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test_acc: {}".format(test_acc))

