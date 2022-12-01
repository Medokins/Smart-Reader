from PIL import Image
import numpy as np
import tensorflow as tf

def imageConverter(name):
    # binarization with very high threshold
    im = Image.open(f'handwrittenNumbers/{name}.png')
    convertedImage = np.zeros((1, 28, 28))

    for x in range(28):
        for y in range(28):
            if(im.getpixel((x,y)) == (255,255,255)):
                convertedImage[0][y][x] = 0.0
            else:
                convertedImage[0][y][x] = 1.0

    return convertedImage

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
    convertedImage = imageConverter(name)
    new_model = tf.keras.models.load_model('numReader.model')
    predictions = new_model.predict([convertedImage])
    print(np.argmax(predictions[0]))