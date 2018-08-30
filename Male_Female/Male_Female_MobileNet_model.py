from __future__ import print_function

import os

import numpy as np
import matplotlib.pyplot as plt
# matplotlib inline

from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNet
from keras.utils import to_categorical
from sklearn import metrics

train_dir = '/Users/mariusdragosionita/PycharmProjects/Image-Classification/Datasets/Male_Female/train'
validation_dir = '/Users/mariusdragosionita/PycharmProjects/Image-Classification/Datasets/Male_Female/validation'

# train_dir = 'C:/Users/Federico/PycharmProjects/Image-Classification/Datasets/fruits/Training/'
# validation_dir = 'C:/Users/Federico/PycharmProjects/Image-Classification/Datasets/fruits/Test/'
image_size = 224

# Load the VGG model
mobNet_conv = MobileNet(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

# Freeze all the layers except last 4
for layer in mobNet_conv.layers[:-4]:
    layer.trainable = False

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(mobNet_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.50))
model.add(layers.Dense(1, activation='sigmoid'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Change the batchsize according to your system RAM
train_batchsize = 10
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

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Train the Model
history = model.fit_generator(
    train_generator,
    epochs=1,
    validation_data=validation_generator,
    verbose=1)

# Save the Model
model.save('left4dead_layers_male_female_data_augmentation.h5')

predictions = model.predict_generator(validation_generator)
print(predictions)
val_preds = [1 if x >= 0.5 else 0 for x in predictions]
print(val_preds)
val_trues = validation_generator.classes
classes_one_hot_encoded = to_categorical(val_trues)

cm = metrics.confusion_matrix(val_trues, val_preds)
print(cm)

precisions, recall, fscore, support = metrics.precision_recall_fscore_support(val_trues, val_preds, average=None)

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

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('tumamma')

plt.show()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('sumamma')

plt.show()