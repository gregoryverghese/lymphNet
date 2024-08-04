#!/usr/bin/env python3

'''
multiscale.py: multiscale unet, uses dilated convolutions.
Uses tensorflow functional API
'''

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Concatenate,concatenate, Conv2DTranspose, ReLU
from tensorflow.keras.layers import  Activation, Dropout, BatchNormalization, MaxPooling2D, Add, Multiply

from .layers import multi_block, conv_block, ConvLayer, UpLayer


class MSUnet():
    def __init__(self, 
                 filters=[32,64,128, 256,512], 
                 final_activation='sigmoid',
                 activation='relu', 
                 n_output=1, 
                 kernel_size=(3,3), 
                 pooling=(2,2), 
                 padding='same', 
                 stride=(1,1),  
                 dilation=1, 
                 initializer='glorot_uniform',
                 dropout=0, 
                 normalize=True,
                 up_type='upsampling', 
                 dtype='float16'):


        self.filters = filters
        self.final_activation = final_activation
        self.activation = activation
        self.n_output = n_output
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.padding = padding
        self.dropout = dropout
        self.n_output = n_output
        self.stride = stride
        self.dilation= dilation
        self.initializer=initializer
        self.dropout=dropout
        self.normalize = normalize
        self.up_type=up_type


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
            self.up_type,
            )
    
    
    def encoder(self, x):

        x = conv_block(x, self.filters[0],self.conv_layer)
        e1 = multi_block(x, self.filters[1],self.conv_layer,self.up_layer)
        e2 = multi_block(e1, self.filters[2], self.conv_layer,self.up_layer)
        e3 = multi_block(e2, self.filters[3], self.conv_layer,self.up_layer)
        e4 = multi_block(e3, self.filters[4], self.conv_layer,self.up_layer)
        e5 = multi_block(e4, self.filters[4], self.conv_layer,self.up_layer)

        return x, e1,e2,e3,e4,e5


    def decoder(self, x, e1,e2,e3,e4,e5):

        d1 = self.up_layer((2, 2))(e5)
        d1 = Concatenate()([e4, d1])
        d1 = conv_block(d1, self.filters[4], self.conv_layer)

        d2 = UpSampling2D((2, 2))(d1)
        d2 = Concatenate()([e3, d2])
        d2 = conv_block(d2, self.filters[3], self.conv_layer)

        d3 = UpSampling2D((2, 2))(d2)
        d3 = Concatenate()([e2, d3])
        d3 = conv_block(d3, self.filters[2], self.conv_layer)

        d4 = UpSampling2D((2, 2))(d3)
        d4 = Concatenate()([e1, d4]) 
        d4 = conv_block(d4, self.filters[1], self.conv_layer)
        
        d5 = upsampling5 = UpSampling2D((2, 2))(d4)
        d5 = Concatenate()([x, d5])
        d5 = conv_block(d5, self.filters[0], self.conv_layer)

        return d5


    def build(self):

        tensor_input = Input((None, None, 3))
        x, e1,e2,e3,e4,e5 = self.encoder(tensor_input)
        x = self.decoder(x, e1,e2,e3,e4,e5)
        final = Conv2D(self.n_output, kernel_size=(1, 1), strides=1,activation=self.final_activation)(x)
        model = Model(tensor_input, final)

        return model
