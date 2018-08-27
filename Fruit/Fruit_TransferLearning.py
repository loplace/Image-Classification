import numpy as np
from keras import models, layers, optimizers
from keras.applications import vgg16
# load vgg model with weights by imagenet
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn import metrics

vgg_conv = vgg16.VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(100, 100, 3))

train_dir = '/home/federico/PycharmProjects/Image Classification/Datasets/fruits/fruits-360/Training'
validation_dir = '/home/federico/PycharmProjects/Image Classification/Datasets/fruits/fruits-360/Test'

nTrain = 37836
nVal = 12709
batch_size = 10
image_size = 100

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
    target_size=(image_size, image_size),
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

validation_datagen = ImageDataGenerator(rescale=1. / 255)

validation_features = np.zeros(shape=(nVal, 7, 7, 512))
validation_labels = np.zeros(shape=nVal)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

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
model.add(layers.Dropout(0.25))
model.add(layers.Dense(75, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features,
                    train_labels,
                    epochs=1,
                    batch_size=batch_size,
                    validation_data=(validation_features, validation_labels))

model.save('Fruit_TransferLearning.h5')


fnames = validation_generator.filenames
ground_truth = validation_generator.classes
label2index = validation_generator.class_indices

predictions = model.predict_classes(validation_features)
prob = model.predict(validation_features)
errors = np.where(predictions != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors), nVal))

val_preds = np.argmax(predictions, axis=-1)
# val_preds = [1 if x >= 0.5 else 0 for x in predictions]
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
