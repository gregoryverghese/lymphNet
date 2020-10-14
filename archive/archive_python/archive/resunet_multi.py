import tensorflow as tf
import keras
import numpy as np


class Unet():
    def __init__(self, imgSize, nOutput=4, activation='relu', finalActivation='sigmoid', padding='same'):
        self.imgSize = imgSize
        self.activation = activation
        self.finalActivation = finalActivation
        self.padding = padding
        self.nOutput = nOutput


    def convBlocks(self, x, filters, kernelSize=(3,3), padding='same', strides=1):

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(self.activation)(x)
        x = keras.layers.Conv2D(filters, kernelSize, padding=padding, strides=strides)(x)

        return x


    def identity(self, x, xInput, f, padding='same', strides=1):

        skip = keras.layers.Conv2D(f, kernel_size=(1, 1), padding=padding, strides=strides)(xInput)
        skip = keras.layers.BatchNormalization()(skip)
        output = keras.layers.Add()([skip, x])

        return output


    def residualBlock(self, xIn, f, stride):

        res = self.convBlocks(xIn, f, strides=stride)
        res = self.convBlocks(res, f, strides=1)
        output = self.identity(res, xIn, f, strides=stride)

        return output


    def upSampling(self, x, xInput):

        x = keras.layers.UpSampling2D((2,2))(x)
        x = keras.layers.Concatenate()([x, xInput])

        return x


    def encoder(self, x, filters, kernelSize=(3,3), padding='same', strides=1):

        e1 = keras.layers.Conv2D(filters[0], kernelSize, padding=padding, strides=strides)(x)
        e1 = self.convBlocks(e1, filters[0])

        shortcut = keras.layers.Conv2D(filters[0], kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = keras.layers.BatchNormalization()(shortcut)
        e1Output = keras.layers.Add()([e1, shortcut])

        e2 = self.residualBlock(e1Output, filters[1], stride=2)
        e3 = self.residualBlock(e2, filters[2], stride=2)
        e4 = self.residualBlock(e3, filters[3], stride=2)
        e5 = self.residualBlock(e4, filters[4], stride=2)

        return e1Output, e2, e3, e4, e5


    def bridge(self, x, filters):

        b1 = self.convBlocks(x, filters, strides=1)
        b2 = self.convBlocks(b1, filters, strides=1)

        return b2


    def decoder(self, b2, e1, e2, e3, e4, filters, kernelSize=(3,3), padding='same', strides=1):

        x = self.upSampling(b2, e4)
        d1 = self.convBlocks(x, filters[4])
        d1 = self.convBlocks(d1, filters[4])
        d1 = self.identity(d1, x, filters[4])

        x = self.upSampling(d1, e3)
        d2 = self.convBlocks(x, filters[3])
        d2 = self.convBlocks(d2, filters[3])
        d2 = self.identity(d2, x, filters[3])

        x = self.upSampling(d2, e2)
        d3 = self.convBlocks(x, filters[2])
        d3 = self.convBlocks(d3, filters[2])
        d3 = self.identity(d3, x, filters[2])

        x = self.upSampling(d3, e1)
        d4 = self.convBlocks(x, filters[1])
        d4 = self.convBlocks(d4, filters[1])
        d4 = self.identity(d4, x, filters[1])

        return d4


    def ResUnet(self, filters = [16, 32, 64, 128, 256]):

        inputs = keras.layers.Input((224, 224, 3))

        e1, e2, e3, e4, e5 = self.encoder(inputs, filters)
        b2 = self.bridge(e5, filters[4])
        d4 = self.decoder(b2, e1, e2, e3, e4, filters)

        x = keras.layers.Conv2D(self.nOutput, (1, 1), padding='same', activation=self.finalActivation)(d4)
        model = keras.models.Model(inputs, x)

        return model
