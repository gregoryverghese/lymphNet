import tensorflow as tf
from tensorflow import keras
import numpy as np

class FCN():
    def __init__(self, imgSize, nClasses, finalActivation):
        self.imgSize = imgSize
        self.nClasses = nClasses
        self.finalActivation = finalActivation


    def convBlock(self, x, filters, kernelSize, activation='relu', padding='same',
                  convName=None, poolName=None, data_format='channels_last'):

        x = keras.layers.Conv2D(filters, kernelSize, activation=activation, padding=padding, name=convName)(x)
        x = keras.layers.Conv2D(filters, kernelSize, activation=activation, padding=padding, name=convName)(x)
        x = keras.layers.MaxPooling2D((2,2), strides=(2,2), name=poolName)(x)

        return x


    def convBlock2(self, x, filters, kernelSize, activation='relu', padding='same',
                  convName=None, poolName=None, data_format='channels_last'):

        x = keras.layers.Conv2D(filters, kernelSize, activation=activation, padding=padding, name=convName)(x)
        x = keras.layers.Conv2D(filters, kernelSize, activation=activation, padding=padding, name=convName)(x)
        x = keras.layers.Conv2D(filters, kernelSize, activation=activation, padding=padding, name=convName)(x)
        x = keras.layers.MaxPooling2D((2,2), strides=(2,2), name=poolName)(x)

        return x


    def encoder(self):

        inputImg = keras.layers.Input(shape=(self.imgSize, self.imgSize, 3))

        pool1 = self.convBlock(inputImg, filters=64, kernelSize=(3,3))
        pool2 = self.convBlock(pool1, filters=128, kernelSize=(3,3))
        pool3 = self.convBlock2(pool2, filters=256, kernelSize=(3,3))
        pool4 = self.convBlock2(pool3, filters=512, kernelSize=(3,3))
        pool5 = self.convBlock2(pool4, filters=512, kernelSize=(3,3))

        conv6 = keras.layers.Conv2D(4096, (7,7), activation='relu', padding='same')(pool5)
        conv7 = keras.layers.Conv2D(4096, (1,1), activation='relu', padding='same')(conv6)

        return conv7, pool4, pool3, inputImg


    def build(self):

        conv7, pool4, pool3, inputImg = self.encoder()
        conv7Up = keras.layers.Conv2DTranspose(self.nClasses, kernel_size=(4,4), strides=(4,4), use_bias=False)(conv7)

        pool4_2 = keras.layers.Conv2D(self.nClasses, (1,1), activation='relu', padding='same')(pool4)
        pool4Up = keras.layers.Conv2DTranspose(self.nClasses, kernel_size=(2,2), strides=(2,2), use_bias=False)(pool4_2)
        skip_7_4 = keras.layers.Add()([pool4Up, conv7Up])

        pool3_2 = keras.layers.Conv2D(self.nClasses, (1,1), activation='relu', padding='same')(pool3)
        skip_7_4_3 = keras.layers.Add()([pool3_2, skip_7_4])

        nOut = 1 if self.nClasses==2 else self.nClasses
        output = keras.layers.Conv2DTranspose(nOut, kernel_size=(8, 8), strides=(8, 8), use_bias=False)(skip_7_4_3)
        output = keras.layers.Activation(self.finalActivation)(output)
        model = keras.models.Model(inputImg, output)

        return model
