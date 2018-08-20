import numpy as np
from keras import models
from sklearn.model_selection import KFold
from sklearn.cross_validation import cross_val_score, cross_val_predict
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

X = np.load('/home/federico/PycharmProjects/Image Classification/SignLanguage/X.npy')
Y = np.load('/home/federico/PycharmProjects/Image Classification/SignLanguage/Y.npy')

batch_size = 16
num_classes = 10
epochs = 1

X = X.reshape(X.shape[0], 64, 64, 1)
input_shape = (64,64,1)

kf = KFold(n_splits=5)  # define number of folds
kf.get_n_splits(X)

print ('kf:')
print (kf)

cvscores = []

for train_index, test_index in kf.split(X):
    print('TRAIN:', train_index, 'TEST:', test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

# define model

    model = None
    model = models.Sequential()

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

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)

    # model.save('mnist_cnn.h5')

    score = model.evaluate(X_test, y_test, verbose=1)
    cvscores.append(score[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

