#!/usr/bin/env python3 

'''
multiscale.py: multiscale unet, uses dilated convolutions
'''

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Concatenate,concatenate, Conv2DTranspose
from tensorflow.keras.layers import  Activation, Dropout, BatchNormalization, MaxPooling2D, Add, Multiply


class ConvBlock(layers.Layer):
    def __init__(self, f, dilation, padding, stride):
        super(ConvBlock, self).__init__()
        self.f = f
        self.batchnorm1 = BatchNormalization()
        self.batchnorm2 = BatchNormalization()
        self.conv2d1 = Conv2D(f, kernel_size=(3,3), activation='relu', strides=stride, padding=padding, dilation_rate=dilation)
        self.conv2d2 = Conv2D(f, kernel_size=(3,3), activation='relu', strides=stride, padding=padding, dilation_rate=dilation)


    def call(self, x, normalize=True):

        x = self.conv2d1(x)
        x = self.batchnorm1(x)
        x = self.conv2d2(x)
        x = self.batchnorm2(x)

        return x


class Up(layers.Layer):
    def __init__(self):
        super(Up, self).__init__()
        self.conv = ConvBlock(outChannel, dilation=1, padding='same', stride=1)

    def call(self, x1, x2):

        upFactor = K.int_shape(d1)[1]/K.int_shape(d2)[1]
        x1 = UpSampling2D((upFactor, upFactor), interpolation='bilinear')
        x1 = Concatenate([x1,x2])

        return self.conv(x1)


class DownBlock(layers.Layer):
    def __init__(self, f):
        super(DownBlock, self).__init__()

        outChannel = f/3

        self.m1 = MaxPooling2D((2,2))

        self.d1 = ConvBlock(outChannel, dilation=1, padding='same', stride=1)

        self.d2 = ConvBlock(outChannel, dilation=2, padding='same', stride=1)
        self.m2 = MaxPooling2D((2,2))

        self.d3 = ConvBlock(outChannel, dilation=4, padding='same', stride=1)
        self.m3 = MaxPooling2D((4,4))

    def call(self, x):

        x = self.m1(x)

        d1 = self.d1(x)

        d2 = self.d2(x)
        d2 = self.m2(d2)
        d2 = UpSampling2D((2,2), interpolation='bilinear')(d2)

        d3 = self.d3(x)
        d3 = self.m3(d3)
        d3 = UpSampling2D((4,4), interpolation='bilinear')(d3)

        out = Concatenate()([d1,d2,d3])

        return out


class MultiScaleUnetSC(Model):
    def __init__(self, filters=[32,64,128, 256,512], finalActivation='sigmoid',
                 activation='relu', nOutput=1, kSize=(3,3), pSize=(2,2),dropout=0, normalize=True, padding='same', upTypeName='upsampling', dtype='float32'):
        super(MultiScaleUnetSC, self).__init__(dtype=dtype)

        self.normalize = normalize
        self.ins = ConvBlock(filters[0], dilation=1, padding='same', stride=1)
        self.convblocke1 = DownBlock(filters[1])
        self.convblocke2 = DownBlock(filters[2])
        self.convblocke3 = DownBlock(filters[3])
        self.convblocke4 = DownBlock(filters[4])
        self.convblocke5 = DownBlock(filters[4])
        
        if upTypeName == 'upsampling':
            self.up1 = UpSampling2D((2, 2))
        elif upTypeName == 'transpose':
            self.up1 = Conv2DTranspose(filters[4], kSize, activation='relu', strides=(2,2), padding='same')

        self.conc1 = Concatenate()
        self.convblockd1 = ConvBlock(filters[4], dilation=1, padding='same', stride=1)

        if upTypeName == 'upsampling':
            self.up2 = UpSampling2D((2, 2))
        elif upTypeName == 'transpose':
            self.up2 = Conv2DTranspose(filters[4], kSize, activation='relu', strides=(2,2), padding='same')

        self.conc2 = Concatenate()
        self.convblockd2 = ConvBlock(filters[3], dilation=1, padding='same', stride=1)

        if upTypeName == 'upsampling':
            self.up3 = UpSampling2D((2, 2))
        elif upTypeName == 'transpose':
            print('TRANSPOSEEEEEEEEEEEEEE')
            self.up3 = Conv2DTranspose(filters[3], kSize, activation='relu',strides=(2,2), padding='same')

        self.conc3 = Concatenate()
        self.convblockd3 = ConvBlock(filters[2], dilation=1, padding='same', stride=1)

        if upTypeName == 'upsampling':
            self.up4 = UpSampling2D((2, 2))
        elif upTypeName == 'transpose':
            self.up4 = Conv2DTranspose(filters[2], kSize, activation='relu',strides=(2,2),padding='same')

        self.conc4 = Concatenate()
        self.convblockd4 = ConvBlock(filters[1], dilation=1, padding='same', stride=1)

        if upTypeName == 'upsampling':
            self.up5 = UpSampling2D((2, 2))
        elif upTypeName == 'transpose':
            self.up5 = Conv2DTranspose(filters[1], kSize,activation='relu',strides=(2,2),padding='same')

        self.conc5 = Concatenate()
        self.convblockd5 = ConvBlock(filters[0], dilation=1, padding='same', stride=1)

        self.final = Conv2D(nOutput, kernel_size=(1, 1), strides=(1, 1), activation=finalActivation)


    def call(self, x, training=True):

        x = self.ins(x)

        e1 = self.convblocke1(x)
        e2 = self.convblocke2(e1)
        e3 = self.convblocke3(e2)
        e4 = self.convblocke4(e3)
        e5 = self.convblocke5(e4)

        d1 = self.up1(e5)
        d1 = self.conc1([e4, d1])
        d1 = self.convblockd1(d1)

        d2 = self.up2(d1)
        d2 = self.conc2([e3, d2])
        d2 = self.convblockd2(d2)

        d3 = self.up3(d2)
        d3 = self.conc3([e2, d3])
        d3 = self.convblockd3(d3)

        d4 = self.up4(d3)
        d4 = self.conc4([e1, d4])
        d4 = self.convblockd4(d4)

        d5 = self.up5(d4)
        d5 = self.conc5([x, d5])
        d5 = self.convblockd5(d5)

        x = self.final(d5)

        return x
