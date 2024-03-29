from __future__ import print_function

import keras
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import precision_recall_fscore_support

from Utilities.Metrics import Metrics

n_epochs = [1, 2, 4, 8, 16, 20]

batch_size = 128
num_classes = 10
# epochs = 12

i = 0
for i in range(0, 1):
    epochs = n_epochs[i]
    print(epochs)
    # sess = K.tensorflow_backend._get_available_gpus()
    # print(sess)

    # input image dimensions
    img_rows, img_cols = 28, 28

    print('downloading mnist data')

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print('downloaded mnist data')

    # change depth of images: if black and white, depth ==1 , if full RGB, depth ==3
    # in this case, depth ==1
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # convert data to Float32 and normalize
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    # Convert 1-dimensional class arrays to 10-dimensional class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    metrics_epoch = Metrics()

    # Next, we declare the input layer: The input shape parameter should be the shape of 1 sample. In this case,
    # it's the same (1, 28, 28) that corresponds to  the (depth, width, height) of each digit image. The first 3
    # parameters represent? They correspond to the number of convolution filters to use, the number of rows in each
    # convolution kernel, and the number of columns in each convolution kernel, respectively.

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))

    # Add another layer
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # MaxPooling2D is a way to reduce the number of parameters in our model by sliding a 2x2 pooling filter across the
    # previous layer and taking the max of the 4 values in the 2x2 filter.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dropout layer we just added. This is a method for regularizing our model in order to prevent overfitting.
    # model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    # Dropout layer we just added. This is a method for regularizing our model in order to prevent overfitting.
    # model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    print('fitting')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[metrics_epoch])

    # model.save('mnist_cnn.h5')
    print('evaluating')
    score = model.evaluate(x_test, y_test, verbose=1)

    # get classification results and transforms into one-hot encoding
    val_predict = model.predict_classes(x_test)
    classes_one_hot_encoded = to_categorical(val_predict)

    # calculate precision, recall, f score, support for each class
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, classes_one_hot_encoded,
                                                                         average='weighted')
    s_precision, s_recall, s_fscore, s_support = precision_recall_fscore_support(y_test, classes_one_hot_encoded,
                                                                                 average=None)

    # metrics given by keras
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # metrics calculated by using sklearn after validating
    print('Precision')
    print(precision)
    print('Recall')
    print(recall)
    print('Fscore')
    print(fscore)

    # metrics calculated by using sklearn after validating
    print('Precision')
    print(s_precision)
    print('Recall')
    print(s_recall)
    print('Fscore')
    print(s_fscore)

    f = open("mnist_cnn.txt", "w+")

    f.write('Weighted Precision:\n')
    str1 = str(precision)
    f.write(str1 + '\n')

    f.write('Weighted Recall:\n')
    str2 = str(recall)
    f.write(str2 + '\n')

    f.write('F_Score:\n')
    str3 = str(fscore)
    f.write(str3 + '\n')

    f.write('val_Acc:\n')
    str3 = str(score[1])
    f.write(str3 + '\n')

    f.write('val_loss:\n')
    str3 = str(score[0])
    f.write(str3 + '\n')

    f.close()
