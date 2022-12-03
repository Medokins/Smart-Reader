import numpy as np
import tensorflow as tf
import cv2
from data_preprocessor import preprocessImage
from sklearn.model_selection import train_test_split
from settings import BATCH_SIZE, EPOCHS

def loadDataSet():
    mnist = tf.keras.datasets.mnist #28x28 images
    ((x_train, y_train), (x_test, y_test)) = mnist.load_data()

    data = np.vstack([x_train, x_test])
    labels = np.hstack([y_train, y_test])

    return (data, labels)

def trainNumReader():
    (digits_data, digits_labels) = loadDataSet()

    digits_data = np.array(digits_data, dtype=np.float32)

    digits_data = np.expand_dims(digits_data, axis=-1)
    digits_data /= 255.0

    (trainX, testX, trainY, testY) = train_test_split(digits_data, digits_labels, test_size=.2, random_state=42)


    model = tf.keras.models.Sequential()
    # takes our 28x28 and makes it 1x784
    model.add(tf.keras.layers.Flatten())
    # fully-connected layer, 128 units, relu activation
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
    # fully-connected layer, 128 units, relu activation
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))
    
    aug = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.05,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=False,
        fill_mode="nearest")

    # training
    model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    model.fit(
        aug.flow(trainX, trainY, batch_size = BATCH_SIZE),
        validation_data = (testX, testY),
        steps_per_epoch = len(trainX) // BATCH_SIZE,
        epochs=EPOCHS)

    # # evaluation
    val_loss, val_acc = model.evaluate(testX, testY)
    print(val_loss, val_acc)
    model.save('newModel.model')

def readDigit(name):
    im = cv2.imread(f'handwrittenNumbers/{name}.png', cv2.IMREAD_GRAYSCALE)
    convertedImage = preprocessImage(im)
    new_model = tf.keras.models.load_model('newModel.model')
    predictions = new_model.predict([convertedImage])
    print(np.argmax(predictions[0]))