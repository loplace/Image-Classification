from __future__ import print_function

# matplotlib inline
from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn import metrics

from Utilities.Metrics import Metrics, MetricsWithGenerator

# train_dir = '/home/federico/PycharmProjects/Image Classification/Datasets/cat-and-dog/training_set'
# validation_dir = '/home/federico/PycharmProjects/Image Classification/Datasets/cat-and-dog/test_set'

train_dir = 'C:/Users/Federico/PycharmProjects/Image-Classification/Datasets/cats-dogs/training_set/'
validation_dir = 'C:/Users/Federico/PycharmProjects/Image-Classification/Datasets/cats-dogs/test_set/'
image_size = 200

# Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
epochs = 10


# Freeze all the layers except last 4
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

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
train_batchsize = 50
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

metrics_epoch = MetricsWithGenerator(validation_generator)
# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adadelta(),
              metrics=['acc'])

# Train the Model
history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[metrics_epoch],
    verbose=1)

# Save the Model
model.save('Dogs_Cats_Finetuning.h5')

predictions = model.predict_generator(validation_generator)
val_preds = [1 if x >= 0.5 else 0 for x in predictions]
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


f = open("Dogs_Cats_FineTuning.txt", "w+")

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