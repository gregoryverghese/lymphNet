#!/usr/bin/env python3

'''
atten_unet.py: attention unet model in both functional and subclass forms
inspired by 'Attention U-Net: Learning Where to Look for the Pancreas' oktay et.al.
'''

import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2,l1_l2
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Concatenate, concatenate, Conv2DTranspose, ReLU
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, MaxPooling2D, Add, Multiply, Input


#################################### subclassing model ####################################

class ConvBlock(layers.Layer):
    def __init__(self, f, dropout, kSize, dtype):
        super(ConvBlock, self).__init__()

        self.conv1 = Conv2D(f, kSize, activation='relu', padding='same')
        self.batchnorm1 = BatchNormalization()
        self.dropout1 = Dropout(dropout)
        self.conv2 = Conv2D(f, kSize, activation='relu', padding='same')
        self.batchnorm2 = BatchNormalization()
        self.dropout2 = Dropout(dropout)


    def call(self, x, normalize):

        x = self.conv1(x)
        x = self.batchnorm1(x) if normalize else x
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x) if normalize else x
        x = self.dropout2(x)

        return x


class AttentionBlock(layers.Layer):
    def __init__(self, f1, f2):
        super(AttentionBlock, self).__init__()
        self.theta = Conv2D(f1, kernel_size=(2, 2), strides=(2, 2), padding ='same', use_bias=False)
        self.phi = Conv2D(f1, kernel_size=(1,1), strides=1, padding='same', use_bias=True)
        self.psi = Conv2D(1, kernel_size=(1,1), strides=1, padding='same', use_bias=True)
        self.add = Add()
        self.activation1 = Activation(activation='relu')
        self.activation2 = Activation(activation='sigmoid')
        self.multiply = Multiply()
        self.w = Conv2D(f2, kernel_size=1, strides=1, padding='same')
        self.batchnorm = BatchNormalization()


    def call(self, x, g, normalize):

        thetaX = self.theta(x)
        phiG = self.phi(g)
        upFactor1 = K.int_shape(thetaX)[1]/K.int_shape(phiG)[1]
        phiG = UpSampling2D(size=(int(upFactor1), int(upFactor1)), interpolation='bilinear')(phiG)
        psi = self.psi(self.add([phiG, thetaX]))
        psi = self.activation1(psi)
        psi = self.activation2(psi)
        upFactor2 = K.int_shape(x)[1]/K.int_shape(psi)[1]
        psi = UpSampling2D(size=(int(upFactor2), int(upFactor2)), interpolation='bilinear')(psi)
        psi = self.multiply([x, psi])
        psi = self.w(psi)
        psi = self.batchnorm(psi) if normalize else psi

        return psi


class GridGatingSignal(layers.Layer):
    def __init__(self, f):
        super(GridGatingSignal, self).__init__()

        self.conv1 = Conv2D(f, kernel_size=(1,1),strides=(1,1), padding='same')
        self.batchnorm1 = BatchNormalization()
        self.activation1 = Activation(activation='relu')

    def call(self, x, normalize=True):

        x = self.conv1(x)
        x = self.batchnorm1(x) if normalize else x
        x = self.activation1(x)

        return x


class AttenUnetSC(Model):
    def __init__(self, filters=[16,32,64,128,256], finalActivation='sigmoid', activation='relu', kSize=(3,3),
                 nOutput=1, padding='same', dropout=0, upTypeName='upsampling', dilation=(1,1), normalize=True, dtype='float32'):
        super(AttenUnetSC, self).__init__()
        self.upTypeName = upTypeName
        self.normalize = normalize
        self.convblocke1 = ConvBlock(filters[0], dropout, kSize, dtype)
        self.pool1 = MaxPooling2D(pool_size=(2,2))
        self.convblocke2 = ConvBlock(filters[1], dropout, kSize, dtype)
        self.pool2 = MaxPooling2D(pool_size=(2,2))
        self.convblocke3 = ConvBlock(filters[2], dropout, kSize, dtype)
        self.pool3 = MaxPooling2D(pool_size=(2,2))
        self.convblocke4 = ConvBlock(filters[3], dropout, kSize, dtype)
        self.pool4 = MaxPooling2D(pool_size=(2,2))
        self.convblocke5 = ConvBlock(filters[4], dropout, kSize, dtype)

        self.gridGating = GridGatingSignal(filters[3])
         
        if self.upTypeName == 'upsampling':
            self.up1 = UpSampling2D((2, 2))
        elif self.upTypeName == 'transpose':
            self.up1  = Conv2DTranspose(filters[4], kSize, activation='relu', strides=(2,2), padding='same')
        
        self.conc1 = Concatenate()
        self.convblockd1 = ConvBlock(filters[3], dropout, kSize, dtype)
 
        if self.upTypeName == 'upsampling':
            self.up2 = UpSampling2D((2, 2))
        elif self.upTypeName == 'transpose':
            self.up2  = Conv2DTranspose(filters[3], kSize, activation='relu', strides=(2,2), padding='same')

        self.conc2 = Concatenate()
        self.convblockd2 = ConvBlock(filters[2], dropout, kSize, dtype)

        if self.upTypeName == 'upsampling':
            self.up3 = UpSampling2D((2, 2))
        elif self.upTypeName == 'transpose':
            print('using transpose')
            self.up3 = Conv2DTranspose(filters[2], kSize, activation='relu', strides=(2,2), padding='same')

        self.conc3 = Concatenate()
        self.convblockd3 = ConvBlock(filters[1], dropout, kSize, dtype)
         
        if self.upTypeName == 'upsampling':
            self.up4 = UpSampling2D((2, 2))
        elif self.upTypeName == 'transpose':
            self.up4 = Conv2DTranspose(filters[1], kSize, activation='relu', strides=(2,2), padding='same')

        self.conc4 = Concatenate()
        self.convblockd4 = ConvBlock(filters[0], dropout, kSize, dtype)
 
        self.attention2 = AttentionBlock(filters[1], filters[1])
        self.attention3 = AttentionBlock(filters[2], filters[2])
        self.attention4 = AttentionBlock(filters[3], filters[3])

        self.finalconv = Conv2D(nOutput, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')

    def call(self, x):

        e1 = self.convblocke1(x, self.normalize)
        p1 = self.pool1(e1)
        e2 = self.convblocke2(p1, self.normalize)
        p2 = self.pool2(e2)
        e3 = self.convblocke3(p2, self.normalize)
        p3 = self.pool3(e3)
        e4 = self.convblocke4(p3, self.normalize)
        p4 = self.pool4(e4)

        bridge = self.convblocke5(p4, self.normalize)
        gating = self.gridGating(bridge)

        a4 = self.attention4(e4, gating, self.normalize)
        a3 = self.attention3(e3, gating, self.normalize)
        a2 = self.attention2(e2, gating, self.normalize)

        d4 = self.up1(bridge)
        d4 = self.conc1([d4, a4])
        d4 = self.convblockd1(d4, self.normalize)

        d3 = self.up2(d4)
        d3 = self.conc2([d3, a3])
        d3 = self.convblockd2(d3, self.normalize)

        d2 = self.up3(d3)
        d2 = self.conc3([d2, a2])
        d2 = self.convblockd3(d2, self.normalize)

        d1 = self.up4(d2)
        d1 = self.conc4([d1, e1])
        d1 = self.convblockd4(d1, self.normalize)

        x = self.finalconv(d1)

        return x 


#################################### functional model ####################################


class AttenUnetFunc():
    def __init__(self, filters=[16,32,64,128,256], finalActivation='sigmoid', 
                 activation='relu', kSize=(3,3), nOutput=1, padding='same', dropout=0, 
                 dilation=(1,1), normalize=True, upTypeName='upsampling', dtype='float32'):

        self.filters = filters
        self.activation = activation
        self.finalActivation = finalActivation
        self.padding = padding
        self.normalize = normalize
        self.kernelSize = kSize
        self.nOutput = nOutput
        self.dropout = dropout
        self.upTypeName = upTypeName


    def convBlock(self, x, f, contraction=True):


        x = Conv2D(filters=f, kernel_size=self.kernelSize,padding=self.padding,
                   kernel_initializer='glorot_uniform')(x)
        x = BatchNormalization()(x) if self.normalize else x
        x = ReLU()(x)
        #x = Dropout(0.1)(x) if contraction else x
        x = Conv2D(filters=f, kernel_size=self.kernelSize,padding=self.padding,
                  kernel_initializer='glorot_uniform')(x)
        x = BatchNormalization()(x) if self.normalize else x
        x = ReLU()(x)
        #x = Dropout(0.1)(x) if contraction else x

        return x


    def attention(self, x, g, f1, f2, u):

        thetaX = Conv2D(f1, kernel_size=(2,2), strides=(2,2), padding='same', use_bias=False)(x)
        phiG = Conv2D(f1, kernel_size=(1,1),strides=1, padding='same', use_bias=True)(g)

        #hardcoding here since I don't define input shape
        #upFactor = K.int_shape(thetaX)[1]/K.int_shape(phiG)[1]
        upFactor = u

        phiG = UpSampling2D(size=(int(upFactor), int(upFactor)), interpolation='bilinear')(phiG)
        psi = Conv2D(1, kernel_size=(1,1), strides=1, padding='same', use_bias=True)(Add()([phiG, thetaX]))
        psi = Activation(activation='relu')(psi)
        psi = Activation(activation='sigmoid')(psi)

        #hardcoding here to
        #upFactor = K.int_shape(x)[1]/K.int_shape(psi)[1]
        upFactor = 2

        psi = UpSampling2D(size=(int(upFactor), int(upFactor)), interpolation='bilinear')(psi)
        psi = Multiply()([x, psi])
        psi = Conv2D(f2, kernel_size=(1,1), strides=(1,1), padding='same')(psi)
        psi = BatchNormalization()(psi)

        return psi


    def gridGatingSignal(self, bridge, f):

        x = Conv2D(f, kernel_size=(1,1),strides=(1,1), padding='same')(bridge)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)

        return x


    def encoder(self, x):

        e1 = self.convBlock(x, self.filters[0])
        p1 = MaxPooling2D(pool_size=(2,2))(e1)
        e2 = self.convBlock(p1, self.filters[1])
        p2 = MaxPooling2D(pool_size=(2,2))(e2)
        e3 = self.convBlock(p2, self.filters[2])
        p3 = MaxPooling2D(pool_size=(2,2))(e3)
        e4 = self.convBlock(p3, self.filters[3])
        p4 = MaxPooling2D(pool_size=(2,2))(e4)
        bridge = self.convBlock(p4, self.filters[4])

        return e1,e2,e3,e4,bridge



    def decoder(self, e1, e2, e3, e4, bridge):

        gating = self.gridGatingSignal(bridge, self.filters[3])
        
        a4 = self.attention(e4, gating, self.filters[3], self.filters[3], 1)
        a3 = self.attention(e3, gating, self.filters[2], self.filters[2], 2)
        a2 = self.attention(e2, gating, self.filters[1], self.filters[1], 4)

        if self.upTypeName == 'upsampling':
            d4 = UpSampling2D((2, 2))(bridge)
        elif self.upTypeName == 'transpose':
            d4 = Conv2DTranspose(filters[4], kSize, activation='relu', stride=(2,2), padding='same')

        d4 = Concatenate()([d4, a4])
        d4 = self.convBlock(d4, self.filters[3])

        if self.upTypeName == 'upsampling':
            d3 = UpSampling2D((2, 2))(d4)
        elif self.upTypeName == 'transpose':
            d3 = Conv2DTranspose(filters[4], kSize, activation='relu', stride=(2,2), padding='same')

        d3 = Concatenate()([d3, a3])
        d3 = self.convBlock(d3, self.filters[2])

        if self.upTypeName == 'upsampling':
            d2 = UpSampling2D((2, 2))(d3)
        elif self.upTypeName == 'transpose':
            d2 = Conv2DTranspose(filters[4], kSize, activation='relu', stride=(2,2), padding='same')

        d2 = Concatenate()([d2, a2])
        d2 = self.convBlock(d2, self.filters[1])

        if self.upTypeName == 'upsampling':
            d1 = UpSampling2D((2, 2))(d2)
        elif self.upTypeName == 'transpose':
            d1 = Conv2DTranspose(filters[4], kSize, activation='relu', stride=(2,2), padding='same')

        d1 = Concatenate()([d1, e1])
        d1 = self.convBlock(d1, self.filters[0])

        return d1


    def build(self):

        tensorInput = Input((None, None, 3))
        e1,e2,e3,e4,bridge = self.encoder(tensorInput)
        d = self.decoder(e1,e2,e3,e4,bridge)
        finalMap = Conv2D(self.nOutput, kernel_size=(1,1), strides=(1, 1), activation=self.finalActivation)(d)
        x = Model(tensorInput, finalMap)

        return x
