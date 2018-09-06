
from __future__ import print_function

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from sklearn import metrics

from Utilities.Metrics import Metrics, MetricsWithGenerator

batch_size = 30
num_classes = 10
epochs = 10

# input image dimensions
image_size = 200

train_dir = 'C:/Users/Federico/PycharmProjects/Image-Classification/Datasets/cats-dogs/training_set/'
validation_dir = 'C:/Users/Federico/PycharmProjects/Image-Classification/Datasets/cats-dogs/test_set/'

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Change the batchsize according to your system RAM
train_batchsize = 30
val_batchsize = 10

# Data Generator for Training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=train_batchsize,
    class_mode='binary')

# Data Generator for Validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=val_batchsize,
    class_mode='binary',
    shuffle=False)

model = Sequential()
metrics_epoch = MetricsWithGenerator(validation_generator)

# Next, we declare the input layer: The input shape parameter should be the shape of 1 sample.

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=[200, 200, 3]))

# Add another layer
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
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

print('fitting')
# Train the Model
history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[metrics_epoch])

# model.save('mnist_cnn.h5')
print('evaluating')
score = model.evaluate_generator(validation_generator)

validation_generator.reset()
pred = model.predict_generator(validation_generator, verbose=1)
print(pred)
predicted_class_indices = [1 if x >= 0.5 else 0 for x in pred]

print(predicted_class_indices)
labels = train_generator.class_indices
print(labels)
labels = dict((v, k) for k, v in labels.items())
print(labels)
predictions = [labels[k] for k in predicted_class_indices]
print(predictions)
val_trues = validation_generator.classes
print(val_trues)

cm = metrics.confusion_matrix(val_trues, predicted_class_indices)
print(cm)

precisions, recall, fscore, support = metrics.precision_recall_fscore_support(val_trues, predicted_class_indices)

# Plot the accuracy and loss curves
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# metrics calculated by using sklearn after validating
print('Precision')
print(precisions)
print('Recall')
print(recall)
print('Fscore')
print(fscore)

f = open("Dogs_Cats_Scratch.txt", "w+")

f.write('Number of Epochs:' + str(epochs) + '\n')

f.write('Weighted Precision:\n')
str1 = str(metrics_epoch.val_precisions)
f.write(str1 + '\n')

f.write('Weighted Recall:\n')
str2 = str(metrics_epoch.val_recalls)
f.write(str2 + '\n')

f.write('F_Score:\n')
str3 = str(metrics_epoch.val_f1s)
f.write(str3 + '\n')

f.write('val_Acc:\n')
str3 = str(metrics_epoch.val_accuracy)
f.write(str3 + '\n')

f.write('val_loss:\n')
str3 = str(metrics_epoch.val_loss)
f.write(str3 + '\n')

f.close()