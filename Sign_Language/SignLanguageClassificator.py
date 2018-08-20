import numpy as np
from sklearn.model_selection import KFold
from sklearn.cross_validation import cross_val_score, cross_val_predict
import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn import metrics

X = np.load('/home/federico/PycharmProjects/Image Classification/SignLanguage/X.npy')
Y = np.load('/home/federico/PycharmProjects/Image Classification/SignLanguage/Y.npy')

batch_size = 128
num_classes = 10
epochs = 1

X = X.reshape(X.shape[0], 64, 64,1)
input_shape = (1,64,64)

kf = KFold(n_splits=5)  # define number of folds
kf.get_n_splits(X)

print ('kf:')
print (kf)

for train_index, test_index in kf.split(X):
    print('TRAIN:', train_index, 'TEST:', test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

# define model

model = Sequential()

# Next, we declare the input layer: The input shape parameter should be the shape of 1 sample. In this case,
# it's the same (1, 28, 28) that corresponds to  the (depth, width, height) of each digit image. The first 3
# parameters represent? They correspond to the number of convolution filters to use, the number of rows in each
# convolution kernel, and the number of columns in each convolution kernel, respectively.

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

# Add anoother layer
model.add(Conv2D(64, (3, 3), activation='relu'))

# MaxPooling2D is a way to reduce the number of parameters in our model by sliding a 2x2 pooling filter across the
# previous layer and taking the max of the 4 values in the 2x2 filter.
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout layer we just added. This is a method for regularizing our model in order to prevent overfitting.
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))

# Dropout layer we just added. This is a method for regularizing our model in order to prevent overfitting.
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X, Y,
          batch_size=batch_size,
          epochs=epochs)

# perform 5-fold cross validation
scores = cross_val_score(model, X, Y, cv=5)

# make crosss validation predictions
predictions = cross_val_predict(model, X, Y, cv=5)
