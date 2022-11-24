# model creation and training
import tensorflow as tf
import numpy as np
from data_processor import  DataProcessor

class Model:
    def __init__(self, data: DataProcessor):
        self.data = data
        self.setup_layers()

    def setup_layers(self):
        #this need to be changed so that x_train containts img array not image path
        (self.x_train, self.y_train) = list(zip(*self.data.train_set))
        (self.x_test, self.y_test) =  list(zip(*self.data.test_set))

        # self.x_train = tf.keras.utils.normalize(self.x_train, axis = 1)
        # self.x_test = tf.keras.utils.normalize(self.x_test, axis = 1)

        # model = tf.keras.models.Sequential()
        # model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
        # model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
        ## here i need to figure out what my output layer should be
        # model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy', metrics=["accuracy"])
        # model.fit(x_train, y_train, epochs = 3)
        # model.save('textReader.model')

def main():
    textReader = Model(DataProcessor(.9))
    print(textReader.x_train[100], textReader.y_train[100])

if __name__ == '__main__':
    main()