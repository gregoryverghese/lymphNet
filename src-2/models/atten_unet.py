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


class AttenUnetFunc():
    def __init__(self, 
                 filters=[32,64,128,256,512], 
                 final_activation='sigmoid', 
                 activation='relu', 
                 kernel_size=(3,3), 
                 n_output=1, 
                 padding='same', 
                 dropout=0, 
                 dilation=(1,1), 
                 normalize=True, 
                 up_layer='upsampling', 
                 dtype='float32'):

        self.filters = filters
        self.activation = activation
        self.final_activation = final_activation
        self.padding = padding
        self.normalize = normalize
        self.kernel_size = kernel_size
        self.n_output = n_output
        self.dropout = dropout
        self.up_layer = up_layer


    @property
    def conv_layer(self):
        return ConvLayer(
             self.kernel_size,
             self.padding,
             self.initializer
             )

    @property
    def up_layer(self):
        return UpLayer(
            self.kernel_size,
            self.padding,
            self.initializer,
            self.activation,
            self.layer_type,
            )


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

        d4 = self.up_layer(self.filters[4])(bridge)
        d4 = Concatenate()([d4, a4])
        d4 = self.conv_block(d4, self.filters[3])

        d3 = self.up_layer(self.filters[3])(d4)
        d3 = Concatenate()([d3, a3])
        d3 = self.conv_block(d3, self.filters[2])

        d2 = self.up_layer(self.filters[2])(d3)
        d2 = Concatenate()([d2, a2])
        d2 = self.conv_block(d2, self.filters[1])

        d1 = self.up_layer(self.filters[1])(d3)
        d1 = Concatenate()([d1, e1])
        d1 = self.convBlock(d1, self.filters[0])

        return d1


    def build(self):

        tensor_input = Input((None, None, 3))
        e1,e2,e3,e4,bridge = self.encoder(tensor_input)
        d = self.decoder(e1,e2,e3,e4,bridge)
        final_map = Conv2D(self.n_output, kernel_size=(1,1), strides=(1, 1),activation=self.final_activation)(d)
        x = Model(tensor_input, final_map)

        return x
