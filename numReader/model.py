import numpy as np
import tensorflow as tf
from data_preprocessor import preprocessImage

def trainNumReader():
    mnist = tf.keras.datasets.mnist #28x28 images
    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis = 1)
    x_test = tf.keras.utils.normalize(x_test, axis = 1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

    model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    model.fit(x_train,y_train, epochs = 5)
    model.save('numReader.model')

def readDigit(name):
    convertedImage = preprocessImage(name)
    new_model = tf.keras.models.load_model('numReader.model')
    predictions = new_model.predict([convertedImage])
    print(np.argmax(predictions[0]))