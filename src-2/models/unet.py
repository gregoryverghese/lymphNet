#!/usr/local/bin/env python3

'''
unet.py: U-Net model using functional tensorflow api
'''

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2,l1
from tensorflow.keras.layers import Conv2D, UpSampling2D,BatchNormalization,GaussianNoise
from tensorflow.keras.layers import MaxPooling2D, Dropout, Activation, Concatenate
from tensorflow.keras.layers import Add, Multiply, Input, Conv2DTranspose,LeakyReLU, ReLU

from .layers import ConvLayer, UpLayer, conv_block


class Unet():
    def __init__(self, 
                 filters=[32,64,128,256,512], 
                 final_activation='sigmoid', 
                 activation='relu',
                 n_output=1, 
                 kernel_size=(3,3), 
                 pool=(2,2),
                 initializer='glorot_uniform',
                 dropout=0, 
                 normalize=True, 
                 padding='same', 
                 up_type='upsampling', 
                 dtype='float32'):

        self.filters = filters
        self.final_activation = final_activation
        self.activation = activation
        self.n_output = n_output
        self.kernel_size = kernel_size
        self.pool = pool
        self.initializer = initializer
        self.dropout=dropout
        self.normalize = normalize
        self.padding = padding
        self.up_type = up_type
        self.dtype = dtype


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
        

    def bridge(self, x, f):

        x = self.conv_layer(f)(x)
        x = BatchNormalization()(x) if self.normalize else x
        x = self.conv_layer(f)(x)
        x = BatchNormalization()(x) if self.normalize else x

        return x


    def encoder(self, x):

        e1 = conv_block(x, self.filters[0], self.conv_layer)
        p1 = MaxPooling2D((2,2))(e1)
        e2 = conv_block(p1, self.filters[1], self.conv_layer)
        p2 = MaxPooling2D((2,2))(e2)
        e3 = conv_block(p2, self.filters[2], self.conv_layer)
        p3 = MaxPooling2D((2,2))(e3)
        e4 = conv_block(p3, self.filters[3], self.conv_layer)
        p4 = MaxPooling2D((2,2))(e4)
        bridge = conv_block(p4, self.filters[4], self.conv_layer)
 
        return e1,e2,e3,e4,bridge


    def decoder(self, e1,e2,e3,e4, bridge):

        d5 = self.up_layer(self.filters[4])(bridge)
        d5 = Concatenate()([e4, d5])
        d5 = conv_block(d5, self.filters[3], self.conv_layer)

        d4 = self.up_layer(self.filters[3])(d5)
        d4 = Concatenate()([e3, d4])
        d4 = conv_block(d4, self.filters[2], self.conv_layer)
        
        d3 = self.up_layer(self.filters[2])(d4)
        d3 = Concatenate()([e2, d3])
        d3 = conv_block(d3, self.filters[1], self.conv_layer)

        d2 = self.up_layer(self.filters[1])(d3)
        d2 = Concatenate()([e1, d2])
        d2 = conv_block(d2, self.filters[0], self.conv_layer)

        return d2


    def build(self):

        input_tensor = Input((None, None, 3))
        e1,e2,e3,e4,bridge = self.encoder(input_tensor)
        d2 = self.decoder(e1,e2,e3,e4, bridge)
        final_tensor = Conv2D(self.n_output, 
                             (1, 1), 
                             strides=(1,1),
                             activation=self.final_activation)(d2)
        '''
        finalMap = CrfRnnLayer(image_dims=(height, width),
                             num_classes=2,
                             theta_alpha=160.,
                             theta_beta=3.,
                             theta_gamma=3.,
                             num_iterations=10,
                             name='crfrnn')([d2,tensorInput])
        '''
        model = Model(input_tensor, final_tensor)
        return model
