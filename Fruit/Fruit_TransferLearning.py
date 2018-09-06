import numpy as np
from keras import models, layers
from keras.applications import vgg16
# load vgg model with weights by imagenet
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator

from Utilities.Metrics import Metrics

vgg_conv = vgg16.VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(100, 100, 3))

vgg_conv.summary()


train_dir = 'C:/Users/Federico/PycharmProjects/Image-Classification/Datasets/fruit/Training'
validation_dir = 'C:/Users/Federico/PycharmProjects/Image-Classification/Datasets/fruit/Test'

nTrain = 38695
nVal = 13000
batch_size = 100
image_size = 100

epochs = 10

datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_features = np.zeros(shape=(nTrain, 3, 3, 512))
train_labels = np.zeros(shape=(nTrain, 77))

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    train_features[i * batch_size: (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nTrain:
        break

train_features = np.reshape(train_features, (nTrain, 3 * 3 * 512))

validation_datagen = ImageDataGenerator(rescale=1. / 255)

validation_features = np.zeros(shape=(nVal, 3, 3, 512))
validation_labels = np.zeros(shape=(nVal,77))

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

i = 0
for inputs_batch, labels_batch in validation_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    validation_features[i * batch_size: (i + 1) * batch_size] = features_batch
    validation_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nVal:
        break

validation_features = np.reshape(validation_features, (nVal, 3 * 3 * 512))

# Generate Model
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_dim=3 * 3 * 512))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(77, activation='softmax'))

model.compile(optimizer='Adadelta',
              loss='categorical_crossentropy',
              metrics=['acc'])

metrics_epoch = Metrics()
history = model.fit(train_features,
                    train_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(validation_features, validation_labels),
                    callbacks=[metrics_epoch])

model.save('Fruit_TransferLearning.h5')

fnames = validation_generator.filenames
ground_truth = validation_generator.classes
label2index = validation_generator.class_indices

predictions = model.predict_classes(validation_features)
prob = model.predict(validation_features)
errors = np.where(predictions != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors), nVal))

val_preds = np.argmax(predictions, axis=-1)
print(val_preds)
val_trues = validation_generator.classes
classes_one_hot_encoded = to_categorical(val_trues)


f = open("Fruit_TransferLearning.txt", "w+")

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