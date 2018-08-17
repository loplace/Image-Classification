# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2



# load the image
image = cv2.imread('/home/federico/Scaricati/mnist_png/testing/1/2.png', cv2.IMREAD_GRAYSCALE)
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (28, 28))
image = image.astype("float32") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model('mnist_cnn.h5')

# classify the input image
result = model.predict(image)[0]
index = result.argmax(axis=-1)
proba = result[index]


print index
print proba

# build the label

label = "{}: {:.2f}%".format(index, proba * 100)

print label

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (0, 0), cv2.FONT_HERSHEY_SIMPLEX,
            700, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
