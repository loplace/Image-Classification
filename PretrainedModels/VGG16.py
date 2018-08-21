import  keras
from keras import models, layers, optimizers
from keras.applications import vgg16,vgg19
import numpy as np

#load vgg model with weights by imagenet
from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

vgg_conv = vgg16.VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

train_dir = '/home/federico/PycharmProjects/Image Classification/Datasets/Male_Female/train'
validation_dir = '/home/federico/PycharmProjects/Image Classification/Datasets/Male_Female/validation'

nTrain = 1945
nVal = 833

datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
batch_size = 20

train_features = np.zeros(shape=(nTrain, 7, 7, 512))
train_labels = np.zeros(shape=(nTrain, 3))

validation_features = np.zeros(shape=(nVal, 7, 7, 512))
validation_labels = np.zeros(shape=(nVal, 2))

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

validation_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    train_features[i * batch_size: (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nTrain:
        break

train_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))

for val_batch, labels_val in validation_generator:
    features_batch = vgg_conv.predict(val_batch)
    validation_features[i * batch_size: (i + 1) * batch_size] = val_batch
    validation_labels[i * batch_size: (i + 1) * batch_size] = labels_val
    i += 1
    if i * batch_size >= nVal:
        break

validation_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))


#Generate Model
# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features,
                    train_labels,
                    epochs=20,
                    batch_size=batch_size,
                    validation_data=(validation_features, validation_labels))