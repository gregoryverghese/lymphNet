import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Concatenate,concatenate, Conv2DTranspose, ReLU
from tensorflow.keras.layers import  Activation, Dropout, BatchNormalization, MaxPooling2D, Add, Multiply
from tensorflow.keras.regularizers import l2

from .layers import ConvLayer, UpLayer, conv_block, multi_block


class MultiAtten():
    def __init__(self, 
                 filters=[32,64,128,256,512],
                 final_activation='sigmoid',
                 activation='relu',
                 n_output=1,
                 kernel_size=(3,3),
                 pool=(2,2),
                 initializer='glorot_uniform',
                 padding='same',
                 stride=1,
                 dilation=1,
                 dropout=0,
                 normalize=True,
                 up_type='upsampling',
                 dtype='float32'):


        self.filters = filters
        self.final_activation = final_activation
        self.activation = activation
        self.padding = padding
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.pool = pool
        self.initializer = initializer
        self.n_output = n_output
        self.stride = stride
        self.dilation = dilation
        self.up_type = up_type


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


    def attention(self, x, g, f1, f2, u):

        theta_x = Conv2D(f1, kernel_size=(2,2), strides=(2,2), padding=self.padding, use_bias=False)(x)
        phi_g = Conv2D(f1, kernel_size=(1,1),strides=1, padding=self.padding, use_bias=True)(g)
        phi_g = BatchNormalization()(phi_g)
        #upFactor = K.int_shape(thetaX)[1]/K.int_shape(phiG)[1]
        up_factor = u
        phi_g = UpSampling2D(size=(int(up_factor), int(up_factor)),interpolation='bilinear')(phi_g)
        psi = Conv2D(1, kernel_size=(1,1), strides=1, padding=self.padding,use_bias=True)(Add()([phi_g, theta_x]))
        psi = BatchNormalization()(psi)
        psi = Activation('relu')(psi)
        psi = Activation('sigmoid')(psi)
        #upFactor = K.int_shape(x)[1]/K.int_shape(psi)[1]
        up_factor = 2
        psi = UpSampling2D(size=(int(up_factor), int(up_factor)), interpolation='bilinear')(psi)
        psi = Multiply()([x, psi])
        psi = Conv2D(f2, kernel_size=(1,1), strides=(1,1), padding=self.padding)(psi)
        psi = BatchNormalization()(psi)

        return psi


    def grid_gating_signal(self, bridge, f):

        x = Conv2D(f, kernel_size=(1,1),strides=(1,1), padding=self.padding)(bridge)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x


    def encoder(self, x):

        x = conv_block(x, self.filters[0],self.conv_layer)
        e1 = multi_block(x, self.filters[1],self.conv_layer,self.up_layer)
        e2 = multi_block(e1, self.filters[2], self.conv_layer,self.up_layer)
        e3 = multi_block(e2, self.filters[3], self.conv_layer,self.up_layer)
        e4 = multi_block(e3, self.filters[4], self.conv_layer,self.up_layer)
        e5 = multi_block(e4, self.filters[4], self.conv_layer,self.up_layer)

        return x,e1,e2,e3,e4,e5


    def decoder(self, x, e1,e2,e3,e4,e5):

        gating = self.grid_gating_signal(e5, self.filters[3])

        a4 = self.attention(e4, gating, self.filters[3], self.filters[3],1)
        a3 = self.attention(e3, gating, self.filters[2], self.filters[2],2)
        a2 = self.attention(e2, gating, self.filters[1], self.filters[1],4)
        a1 = self.attention(e1, gating, self.filters[1], self.filters[1],8)
        
        #print(K.int_shape(d1), K.int_shape(a4))
        d1 = UpSampling2D((2, 2))(e5)

        print(K.int_shape(d1), K.int_shape(a4))
        d1 = Concatenate()([d1, a4])
        d1 = conv_block(d1, self.filters[4], self.conv_layer)

        #print(K.int_shape(d2), K.int_shape(a3))
        d2 = UpSampling2D((2, 2))(d1)

        print(K.int_shape(d2), K.int_shape(a3))
        d2 = Concatenate()([d2, a3])
        d2 = conv_block(d2, self.filters[3], self.conv_layer)


        #print(K.int_shape(d3), K.int_shape(a2))
        d3 = UpSampling2D((2, 2))(d2)

        print(K.int_shape(d3), K.int_shape(a2))
        d3 = Concatenate()([d3, a2])
        d3 = conv_block(d3, self.filters[2], self.conv_layer)

        d4 = UpSampling2D((2, 2))(d3)
        d4 = Concatenate()([d4, a1])
        d4 = conv_block(d4, self.filters[1], self.conv_layer)

        d5 = UpSampling2D((2, 2))(d4)
        d5 = Concatenate()([d5, x])
        d5 = conv_block(d5, self.filters[0], self.conv_layer)

        return d5


    def build(self):

        tensor_input = Input((None, None, 3))
        x, e1,e2,e3,e4,e5 = self.encoder(tensor_input)
        x = self.decoder(x, e1,e2,e3,e4,e5)
        final = Conv2D(self.n_output, kernel_size=(1, 1), strides=1,activation=self.final_activation)(x)
        model = Model(tensor_input, final)

        return model
