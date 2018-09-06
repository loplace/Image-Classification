import keras
from keras import models, layers, optimizers
from keras.applications import vgg16, vgg19
import numpy as np

# load vgg model with weights by imagenet
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn import metrics
from sklearn.utils import shuffle

from Utilities.Metrics import Metrics

vgg_conv = vgg16.VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(224, 224, 3))

train_dir = 'C:/Users/Federico/PycharmProjects/Image-Classification/Datasets/cats-dogs/training_set/'
validation_dir = 'C:/Users/Federico/PycharmProjects/Image-Classification/Datasets/cats-dogs/test_set/'

nTrain = 8005
nVal = 2023
batch_size = 10
epochs = 10

datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_features = np.zeros(shape=(nTrain, 7, 7, 512))
train_labels = np.zeros(shape=nTrain)

train_generator = datagen.flow_from_directory(
    train_dir,
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

validation_features = np.zeros(shape=(nVal, 7, 7, 512))
validation_labels = np.zeros(shape=nVal)

validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)
metrics_epoch = Metrics()

i = 0
for inputs_batch, labels_batch in validation_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    validation_features[i * batch_size: (i + 1) * batch_size] = features_batch
    validation_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nVal:
        break

validation_features = np.reshape(validation_features, (nVal, 7 * 7 * 512))

# Generate Model
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='Adadelta',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features,
                    train_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(validation_features, validation_labels),
                    callbacks=[metrics_epoch]
                    )

fnames = validation_generator.filenames
ground_truth = validation_generator.classes
label2index = validation_generator.class_indices

predictions = model.predict_classes(validation_features)
prob = model.predict(validation_features)
errors = np.where(predictions != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors), nVal))

predicted_class_indices = [1 if x >= 0.5 else 0 for x in prob]
print(predicted_class_indices)
val_trues = validation_generator.classes
classes_one_hot_encoded = to_categorical(val_trues)

cm = metrics.confusion_matrix(val_trues, predicted_class_indices)
print(cm)

precisions, recall, fscore, support = metrics.precision_recall_fscore_support(val_trues, predicted_class_indices,
                                                                              average=None)

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

f = open("Dogs_Cats_TransferLearning.txt", "w+")

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

f.close()
