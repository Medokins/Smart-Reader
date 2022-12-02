import numpy as np
import tensorflow as tf
from data_preprocessor import preprocessImage
import cv2

def trainNumReader():
    mnist = tf.keras.datasets.mnist #28x28 images
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # scales data to be in range 0 1
    x_train = tf.keras.utils.normalize(x_train, axis = 1)
    x_test = tf.keras.utils.normalize(x_test, axis = 1)

    model = tf.keras.models.Sequential()
    # takes our 28x28 and makes it 1x784
    model.add(tf.keras.layers.Flatten())
    # fully-connected layer, 128 units, relu activation
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
    # fully-connected layer, 128 units, relu activation
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

    # training
    model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    model.fit(x_train, y_train, epochs = 3)

    # evaluation
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss, val_acc)
    model.save('numReader.model')

def readDigit(name):
    im = cv2.imread(f'handwrittenNumbers/{name}.png', cv2.IMREAD_GRAYSCALE)
    convertedImage = preprocessImage(im)
    new_model = tf.keras.models.load_model('numReader.model')
    predictions = new_model.predict([convertedImage])
    print(np.argmax(predictions[0]))