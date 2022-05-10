import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Concatenate, concatenate
from tensorflow.keras.layers import  Activation, Dropout, BatchNormalization, MaxPooling2D, Add, Multiply



class ResUnetA():
    def __init__(self, filters=[32,64,128,256,512,1024], finalActivation='sigmoid', activation='relu', kSize=(3,3),
         nOutput=1, padding='same', dropout=0, dilation=(1,1), normalize=True):

        self.filters = filters
        self.activation = activation
        self.finalActivation = finalActivation
        self.padding = padding
        self.normalize = normalize
        self.kernelSize = kSize
        self.nOutput = nOutput
        self.dropout = dropout


    def convBlock(self, x, f, d=1, contraction=True):

        x = Conv2D(filters=f, kernel_size=self.kernelSize, activation=self.activation, dilation_rate=d, padding=self.padding)(x)
        x = BatchNormalization()(x) if self.normalize else x
        x = Dropout(self.dropout)(x) if contraction else x
        x = Conv2D(filters=f, kernel_size=self.kernelSize, activation=self.activation, dilation_rate=d, padding=self.padding)(x)
        x = BatchNormalization()(x) if self.normalize else x
        x = Dropout(self.dropout)(x) if contraction else x

        return x


    def resUnetABlock(self, x, f, dilationRates):

        convBlocks = [self.convBlock(x, f, d) for d in dilationRates]
        maps = [convBlocks[i] for i, d in enumerate(dilationRates)]
        mapfinal = Add()(maps) if len(maps) > 1 else maps[0]

        return mapfinal


    def psPPooling(self, x, f):

        x1 = MaxPooling2D(pool_size=(1,1))(x)
        x2 = MaxPooling2D(pool_size=(2,2))(x)
        x3 = MaxPooling2D(pool_size=(4,4))(x)
        x4 = MaxPooling2D(pool_size=(8,8))(x)

        x1 = Conv2D(int(f/4), kernel_size=(1,1), strides=(1,1), padding='same')(x1)
        x2 = Conv2D(int(f/4), kernel_size=(1,1), strides=(1,1), padding='same')(x2)
        x3 = Conv2D(int(f/4), kernel_size=(1,1), strides=(1,1), padding='same')(x3)
        x4 = Conv2D(int(f/4), kernel_size=(1,1), strides=(1,1), padding='same')(x4)

        x1 = UpSampling2D(size=(1,1))(x1)
        x2 = UpSampling2D(size=(2,2))(x2)
        x3 = UpSampling2D(size=(4,4))(x3)
        x4 = UpSampling2D(size=(8,8))(x4)

        x = Concatenate()([x1,x2,x3,x4])
        x = Conv2D(f, (1,1))(x)

        return x


    def encoder(self, x):

        e1 = Conv2D(self.filters[0], kernel_size=(1, 1), strides=1, padding='same')(x)
        e2 = self.resUnetABlock(e1, f=self.filters[0], dilationRates=[1,3,15,31])
        p2 = Conv2D(self.filters[1], kernel_size=(1, 1), strides=2, padding='same')(e1)
        e3 = self.resUnetABlock(p2, f=self.filters[1], dilationRates=[1,3,15,31])
        p3 = Conv2D(self.filters[2], kernel_size=(1, 1), strides=2, padding='same')(e3)
        e4 = self.resUnetABlock(p3, f=self.filters[2], dilationRates=[1,3,15])
        p4 = Conv2D(self.filters[3], kernel_size=(1, 1), strides=2, padding='same')(e4)
        e5 = self.resUnetABlock(p4, f=self.filters[3], dilationRates=[1,3,15])
        p5 = Conv2D(self.filters[4], kernel_size=(1, 1), strides=2, padding='same')(e5)
        e6 = self.resUnetABlock(p5, f=self.filters[4], dilationRates=[1])
        p6 = Conv2D(self.filters[5], kernel_size=(1, 1), strides=2, padding='same')(e6)
        e7 = self.resUnetABlock(p6, f=self.filters[5], dilationRates=[1])

        return e1,e2,e3,e4,e5,e6,e7


    def decoder(self, e1,e2,e3,e4,e5,e6,p):

        x1 = UpSampling2D((2, 2))(p)
        x1 = Concatenate()([x1, e6])
        x1 = self.resUnetABlock(x1, self.filters[5], dilationRates=[1])

        x2 = UpSampling2D((2, 2))(x1)
        x2 = Concatenate()([x2, e5])
        x2 = self.resUnetABlock(x2, self.filters[4], dilationRates=[1,3,15])

        x3 = UpSampling2D((2, 2))(x2)
        x3 = Concatenate()([x3, e4])
        x3 = self.resUnetABlock(x3, self.filters[3], dilationRates=[1,3,15])

        x4 = UpSampling2D((2, 2))(x3)
        x4 = Concatenate()([x4, e3])
        x4 = self.resUnetABlock(x4, self.filters[2], dilationRates=[1,3,15,31])

        x5 = UpSampling2D((2, 2))(x4)
        x5 = Concatenate()([x5, e2])
        x5 = self.resUnetABlock(x5, self.filters[1], dilationRates=[1,3,15,31])
        x5 = Concatenate()([x5, e1])

        return x5


    def ResUNet(self):

        inputTensor = Input((256,256,3))
        e1,e2,e3,e4,e5,e6,e7 = self.encoder(inputTensor)

        p = self.psPPooling(e7, 1024)

        d = self.decoder(e1,e2,e3,e4,e5,e6,p)
        p = self.psPPooling(d, 32)
        x = Conv2D(32, kernel_size=(1, 1), strides=1, padding='same')(p)
        xFinal = Model(inputTensor, x)

        return xFinal
