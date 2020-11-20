# -*- coding: utf-8 -*-

# <center>Convolutional Neural Networks</center>
## <center>Benedek Dankó</center>
"""

# Commented out IPython magic to ensure Python compatibility.
# load libraries:
import numpy as np
import seaborn as sns
from google.colab import drive
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import confusion_matrix


from tensorflow import keras
from tensorflow.keras.layers import *

# %matplotlib inline

"""### 1. Load the MNIST dataset and create a CNN model"""

# load data:
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train/255 # normalize data to fall between 0-1
print(x_test.shape)
print(x_train.shape)
print(y_test.shape)
print(y_train.shape)

# convert data:
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test,  10)

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

plt.imshow(x_train[0][:,:,0], cmap='gray')
plt.show() # 5

# set up CNN:
cnn_model = keras.models.Sequential()
cnn_model.add(Conv2D(16, (3, 3), input_shape=x_train.shape[1:], padding='valid'))
cnn_model.add(Activation('relu'))
cnn_model.add(Conv2D(16, (3, 3), padding='valid'))
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))

cnn_model.add(Conv2D(32, (3, 3), padding='valid'))
cnn_model.add(Activation('relu'))
cnn_model.add(Conv2D(32, (3, 3), padding='valid'))
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Flatten())

cnn_model.add(Dense(10))
cnn_model.add(Activation('softmax'))

# check model summary:
cnn_model.summary()

"""You can observe the number of the parameters per layer in the last column. <br>
Total number of parameters: 21,498.
"""

# compile model with Adam optimizer, use categorical crossentropy as loss function:
cnn_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# fit model:
cnn_history = cnn_model.fit(x_train, y_train, batch_size=32,
                            epochs=5, validation_data=(x_test, y_test))

print('Accuracy scores: {}'.format(cnn_history.history['val_accuracy']))
print('Categorical cross-entropy loss: {}'.format(cnn_history.history['val_loss']))

"""The final validation accuracy is 99.27%, which is pretty good."""

# perform predictions: 
cnn_predictions = cnn_model.predict(x_test, batch_size=32)

# plot confusion matrix:
conf = confusion_matrix(y_pred=np.argmax(cnn_predictions, 1), y_true=np.argmax(y_test, 1))
plt.figure(figsize=(10, 10))
sns.heatmap(conf, annot=True, cmap='Reds', fmt='g', cbar=False, vmax=30)
plt.title('Confusion matrix', fontsize=17)
plt.xlabel('Predicted label', fontsize=14)
plt.ylabel('Actual label', fontsize=14)
plt.show()

"""After all, the model reached 99.27% accuracy. <br>
As you can see, the model missed a few 5s, and 9s, and predicted them as 3s, and 4s. <br>

Comparing to the fully-connected NN, the CNN predicted for example the 3s and 4s more accurately.

### 2. Download the Street View House Numbers (SVHN) Dataset
"""

# First, download data from http://ufldl.stanford.edu/housenumbers/.

test = loadmat('test_32x32.mat')
train = loadmat('train_32x32.mat')

print(test.keys())
print(train.keys())

# get training, test datasets:
train_x = train['X']
train_y = train['y']

test_x = test['X']
test_y = test['y']

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

# 10 different labels:
set([i[0] for i in list(train_y)])

# unpack the y arrays:
train_y = np.asanyarray([i[0]-1 for i in list(train_y)])
test_y = np.asanyarray([i[0]-1 for i in list(test_y)])

# visualize 5 images randomly:
def show_train_imgs(n=8, m=1):
    for i in range(m):
        for j in range(n):
            idx = np.random.randint(len(y_train))
            plt.subplot(int('1' + str(n) + str(j+1)))
            plt.imshow(np.einsum('klij->jkli', train_x)[idx].astype('int'))
            plt.title('Label: {}'.format(train_y[idx]), fontsize=20)
            plt.axis('off')
        plt.show()

plt.rcParams['figure.figsize'] = (15, 5)
show_train_imgs(5)

# convert data, one-hot encoding of the labels:
train_x = np.einsum('klij->jkli', train_x)
test_x = np.einsum('klij->jkli', test_x)

train_y = keras.utils.to_categorical(train_y, 10)
test_y  = keras.utils.to_categorical(test_y, 10)

# required shapes of the data:
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)



"""We have 10 classes in this dataset. <br>
Furthermore, we have 73257 train examples and 26032 test examples. <br>
Dimension of the images: 32 x 32 pixels with 3 color chanels.

### 3. Train the CNN model seen in the 1st exercise for this dataset
"""

# set up CNN:
cnn_model = keras.models.Sequential()
cnn_model.add(Conv2D(16, (3, 3), input_shape=train_x.shape[1:], padding='valid'))
cnn_model.add(Activation('relu'))
cnn_model.add(Conv2D(16, (3, 3), padding='valid'))
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))

cnn_model.add(Conv2D(32, (3, 3), padding='valid'))
cnn_model.add(Activation('relu'))
cnn_model.add(Conv2D(32, (3, 3), padding='valid'))
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Flatten())

cnn_model.add(Dense(10))
cnn_model.add(Activation('softmax'))

cnn_model.summary()

"""You can observe the number of the parameters per layer in the last column. <br>
Total number of parameters: 24,666.
"""

# compile model with Adam optimizer, use categorical crossentropy as loss function:
cnn_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# fit model:
cnn_history = cnn_model.fit(train_x, train_y, batch_size=32,
                            epochs=15, validation_data=(test_x, test_y))

print('Final accuracy score: {}'.format(cnn_history.history['val_accuracy'][-1]))
print('Final categorical cross-entropy loss: {}'.format(cnn_history.history['val_loss'][-1]))

"""The same CNN model performed worse on this more complex dataset, with 88.31% final validation accuracy (in case of the MNIST dataset: 99.27% accuracy).

### 4. Evaluate performance
"""

# plot training, validation loss:
plt.plot(cnn_history.history['loss'], label='Training loss')
plt.plot(cnn_history.history['val_loss'], label='Validation loss')
plt.legend(loc='upper right')
plt.xlim(0, 14)
plt.title('Training vs. validation loss', fontsize=17)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Categorical cross-entropy loss', fontsize=14)
plt.grid()
plt.show()

# plot training, validation accuracy:
plt.plot(cnn_history.history['accuracy'], label='Training accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='Validation accuracy')
plt.legend(loc='lower right')
plt.xlim(0, 14)
plt.title('Training vs. validation accuracy', fontsize=17)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid()
plt.show()

"""It seems that we overfit, since the training loss is decreasing, but the validation loss starts to oscillate/to increase. 
However, if the validation accuracy is still increasing, then I guess it is OK.
"""

# perform predictions: 
cnn_predictions = cnn_model.predict(test_x, batch_size=32)

# plot confusion matrix:
conf = confusion_matrix(y_pred=np.argmax(cnn_predictions, 1), y_true=np.argmax(test_y, 1))
plt.figure(figsize=(10, 10))
sns.heatmap(conf, annot=True, cmap='Reds', fmt='g', cbar=False, vmax=500)
plt.title('Confusion matrix', fontsize=17)
plt.xlabel('Predicted label', fontsize=14)
plt.ylabel('Actual label', fontsize=14)
plt.show()

"""The model missed sometimes (> 100) the pictures with labels 0 , 2, 3, 5, and 6.

### 5. Train an other CNN
"""

# build CNN:
cnn_model = keras.models.Sequential()
cnn_model.add(Conv2D(16, (3, 3), input_shape=train_x.shape[1:], padding='same'))
cnn_model.add(Activation('relu'))
cnn_model.add(Conv2D(16, (3, 3), padding='same'))
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))

cnn_model.add(Conv2D(32, (3, 3), padding='same'))
cnn_model.add(Activation('relu'))
cnn_model.add(Conv2D(32, (3, 3), padding='same'))
cnn_model.add(Activation('relu'))
cnn_model.add(Conv2D(32, (3, 3), padding='same'))
cnn_model.add(Activation('relu'))
cnn_model.add(Conv2D(32, (3, 3), padding='same'))
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Conv2D(32, (3, 3), padding='same'))
cnn_model.add(Activation('relu'))
cnn_model.add(Conv2D(32, (3, 3), padding='same'))
cnn_model.add(Activation('relu'))
cnn_model.add(Conv2D(32, (3, 3), padding='same'))
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Conv2D(32, (3, 3), padding='same'))
cnn_model.add(Activation('relu'))
cnn_model.add(Conv2D(32, (3, 3), padding='same'))
cnn_model.add(Activation('relu'))
cnn_model.add(Flatten())

cnn_model.add(Dense(10))
cnn_model.add(Activation('softmax'))

# compile model with Adam optimizer, use categorical crossentropy as loss function:
cnn_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# fit model:
cnn_history = cnn_model.fit(train_x, train_y, batch_size=32,
                            epochs=10, validation_data=(test_x, test_y))

cnn_model.summary()

"""The model has 86,552 parameters in total."""

# plot training, validation loss:
plt.plot(cnn_history.history['loss'], label='Training loss')
plt.plot(cnn_history.history['val_loss'], label='Validation loss')
plt.legend(loc='right')
plt.xlim(0, 9)
plt.title('Training vs. validation loss', fontsize=17)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Categorical cross-entropy loss', fontsize=14)
plt.grid()
plt.show()

# plot training, validation accuracy:
plt.plot(cnn_history.history['accuracy'], label='Training accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='Validation accuracy')
plt.legend(loc='lower right')
plt.xlim(0, 9)
plt.title('Training vs. validation accuracy', fontsize=17)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid()
plt.show()

"""As we can see, the model's accuracy reached it's maximum value in the 10th epoch (for both validation and train data). <br>
However, the lowest validation loss value is reached in the 6th epoch.
"""