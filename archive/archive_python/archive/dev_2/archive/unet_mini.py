import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Dropout, Conv2D, Add
from tensorflow.keras.layers import  MaxPooling2D, UpSampling2D, Activation, Concatenate


class ConvBlock(layers.Layer):
    def __init__(self, f, dtype='float16', kSize=(3,3), drop=0):
        super(ConvBlock, self).__init__(dtype=dtype)
        self.f = f
        self.batchnorm = BatchNormalization()
        self.dropout = Dropout(0)
        self.conv2d1 = Conv2D(f, kSize, activation='relu', padding='same')
        self.conv2d2 = Conv2D(f, kSize, activation='relu', padding='same')


    def call(self, x):
        x = self.conv2d1(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.conv2d2(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        return x


class Bridge(layers.Layer):
    def __init__(self, f, dtype='float16'):
        super(Bridge, self).__init__(dtype=dtype)
        self.conv2d1 = Conv2D(f, (3, 3), activation='relu', padding='same')
        self.batchnorm = BatchNormalization()
        self.conv2d2 = Conv2D(f, (3, 3), activation='relu', padding='same')

    def call(self, x):
        x = self.conv2d1(x)
        x = self.batchnorm(x)
        x = self.conv2d2(x)
        x = self.batchnorm(x)
        return x


class Encoder(layers.Layer):
    def __init__(self, dtype='float16'):
        super(Encoder, self).__init__(dtype=dtype)
        self.convblock1 = ConvBlock(16)
        self.convblock2 = ConvBlock(32)
        self.convblock3 = ConvBlock(64)
        self.pooling = MaxPooling2D((2, 2))
        self.bridge = Bridge(128)


    def call(self, inputTensor):
        x1 = self.convblock1(inputTensor)
        p1 = self.pooling(x1)
        x2 = self.convblock2(p1)
        p2 = self.pooling(x2)
        x3 = self.convblock3(p2)
        p3 = self.pooling(x3)
        b1 = self.bridge(p3)

        return b1,x3,x2,x1


class Decoder(layers.Layer):
    def __init__(self, dtype='float16'):
        super(Decoder, self).__init__(dtype=dtype)
        self.upsampling = UpSampling2D((2, 2))
        self.convblock1 = ConvBlock(64)
        self.convblock2 = ConvBlock(32)
        self.convblock3 = ConvBlock(16)


    def call(self, b1,e3,e2,e1):

        x3 = self.upsampling(b1)
        x3 = Concatenate()([e3, x3])
        x3 = self.convblock1(x3)
        
        x2 = self.upsampling(x3)
        x2 = Concatenate()([e2, x2])
        x2 = self.convblock2(x2)

        x1 = self.upsampling(x2)
        x1 = Concatenate()([e1, x1])
        x1 = self.convblock3(x1)

        return x1


class UnetMini(Model):
    def __init__(self, nOutput, finalActivation, dtype='float16'):
        super(UnetMini, self).__init__(dtype=dtype)
        tf.keras.backend.set_floatx('float16')
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.final = Conv2D(nOutput, kernel_size=(1, 1), strides=(1, 1), activation=finalActivation)

    def call(self, x):
        b1,e3,e2,e1 = self.encoder(x)
        x = self.decoder(b1,e3,e2,e1)
        x = self.final(x)
        return x
