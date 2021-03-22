#!/usr/bin/env python3

'''
multiscale.py: multiscale unet, uses dilated convolutions
'''

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Concatenate,concatenate, Conv2DTranspose, ReLU
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
                 activation='relu', nOutput=1, kSize=(3,3), pSize=(2,2),dropout=0,
                 normalize=True, padding='same', upTypeName='upsampling', dtype='float32'):
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


class MultiScaleUnetFunc():
    def __init__(self, filters=[32,64,128, 256,512], finalActivation='sigmoid',
                 activation='relu', nOutput=1, kSize=(3,3), pSize=(2,2), pool=(1,1),
                 padding='same', stride=1,  dilation=1, dropout=0, normalize=True,
                 upTypeName='upsampling', dtype='float32'):

        self.normalize = normalize
        self.filters = filters
        self.finalActivation = finalActivation
        self.padding = padding
        self.dropout = dropout
        self.kSize = kSize
        self.pSize = pSize
        self.nOutput = nOutput
        self.stride = stride
        self.pool = pool
        self.dilation =dilation


    def convBlock(self, x, f, dilation=1, pool=(1,1)):

        x = Conv2D(f, kernel_size=(3,3), strides=self.stride, padding=self.padding, dilation_rate=dilation)(x)
        x = MaxPooling2D((pool))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        #x = Dropout(0.2)(x)
        x = Conv2D(f, kernel_size=(3,3), strides=self.stride, padding=self.padding, dilation_rate=dilation)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        #x = Dropout(0.2)(x)

        return x


    def downBlock(self, x, f):

        outChannel = f/3

        x = MaxPooling2D((2,2))(x)
        d1 = self.convBlock(x, outChannel, dilation=self.dilation, pool=self.pool)

        d2 = self.convBlock(x, outChannel, dilation=2, pool=(2,2))
        d2 = UpSampling2D((2,2), interpolation='bilinear')(d2)

        d3 = self.convBlock(x, outChannel, dilation=4, pool=(4,4))
        d3 = UpSampling2D((4,4), interpolation='bilinear')(d3)

        out = Concatenate()([d1,d2,d3])

        return out


    def encoder(self, x):

        x = self.convBlock(x, self.filters[0])
        e1 = self.downBlock(x, self.filters[1])
        e2 = self.downBlock(e1, self.filters[2])
        e3 = self.downBlock(e2, self.filters[3])
        e4 = self.downBlock(e3, self.filters[4])
        e5 = self.downBlock(e4, self.filters[4])

        return x, e1,e2,e3,e4,e5


    def decoder(self, x, e1,e2,e3,e4,e5):

        d1 = upsampling1 = UpSampling2D((2, 2))(e5)
        d1 = Concatenate()([e4, d1])
        d1 = self.convBlock(d1, self.filters[4], dilation=1)

        d2 = UpSampling2D((2, 2))(d1)
        d2 = Concatenate()([e3, d2])
        d2 = self.convBlock(d2, self.filters[3], dilation=1)

        d3 = UpSampling2D((2, 2))(d2)
        d3 = Concatenate()([e2, d3])
        d3 = self.convBlock(d3, self.filters[2], dilation=1)

        d4 = UpSampling2D((2, 2))(d3)
        d4 = Concatenate()([e1, d4])
        d4 = self.convBlock(d4, self.filters[1], dilation=1)

        d5 = upsampling5 = UpSampling2D((2, 2))(d4)
        d5 = Concatenate()([x, d5])
        d5 = self.convBlock(d5, self.filters[0], dilation=1)

        return d5


    def build(self):

        tensorInput = Input((None, None, 3))
        x, e1,e2,e3,e4,e5 = self.encoder(tensorInput)
        x = self.decoder(x, e1,e2,e3,e4,e5)
        final = Conv2D(self.nOutput, kernel_size=(1, 1), strides=1, activation=self.finalActivation)(x)
        model = Model(tensorInput, final)

        return model
