import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Concatenate, concatenate
from tensorflow.keras.layers import  Activation, Dropout, BatchNormalization, MaxPooling2D, Add, Multiply



#################################### subclassing model ####################################

class ConvBlock(layers.Layer):
    def __init__(self, f, dropout, kSize, dtype):
        super(ConvBlock, self).__init__()
        self.f = f
        self.dropout = Dropout(dropout)
        self.batchnorm = BatchNormalization()
        self.conv2d1 = Conv2D(f, kSize, activation='relu', padding='same')
        self.conv2d2 = Conv2D(f, kSize, activation='relu', padding='same')


    def call(self, x, normalize):
        x = self.conv2d1(x)
        x = self.batchnorm(x) if normalize else x
        x = self.dropout(x)
        x = self.conv2d2(x)
        x = self.batchnorm(x) if normalize else x
        x = self.dropout(x)
        return x


class AttentionBlock(layers.Layer):
    def __init__(self, f1, f2):
        super(AttentionBlock, self).__init__()
        self.theta = Conv2D(f1, kernel_size=(2, 2), strides=(2, 2), padding ='same', use_bias=False)
        self.phi = Conv2D(f1, kernel_size=(1,1), strides=1, padding='same', use_bias=True)
        self.psi = Conv2D(1, kernel_size=(1,1), strides=1, padding='same', use_bias=True)
        self.w = Conv2D(f2, kernel_size=1, strides=1, padding='same')


    def call(self, x, g, normalize):

        thetaX = self.theta(x)
        phiG = self.phi(g)
        upFactor = K.int_shape(thetaX)[1]/K.int_shape(phiG)[1]
        phiG = UpSampling2D(size=(upFactor, upFactor), interpolation='bilinear')(phiG)
        psi = self.psi(Add()([phiG, thetaX]))
        psi = Activation(activation='relu')(psi)
        psi = Activation(activation='sigmoid')(psi)
        upFactor = K.int_shape(x)[1]/K.int_shape(psi)[1]
        psi = UpSampling2D(size=(upFactor, upFactor), interpolation='bilinear')(psi)
        psi = Multiply()([x, psi])
        psi = self.w(psi)
        psi = BatchNormalization()(psi) if normalize else psi

        return psi


class GridGatingSignal(layers.Layer):
    def __init__(self, f):
        super(GridGatingSignal, self).__init__()
        self.conv1 = Conv2D(f, kernel_size=(1,1),strides=(1,1), padding='same')


    def call(self, x, normalize=True):

        x = self.conv1(x)
        x = BatchNormalization()(x) if normalize else x
        x = Activation(activation='relu')(x)

        return x

class Encoder(layers.Layer):
    def __init__(self, filters, dropout, normalize, kSize, dtype):
        super(Encoder, self).__init__()
        self.normalize = normalize
        self.convBlock1 = ConvBlock(filters[0], dropout, kSize, dtype)
        self.convBlock2 = ConvBlock(filters[1], dropout, kSize, dtype)
        self.convBlock3 = ConvBlock(filters[2], dropout, kSize, dtype)
        self.convBlock4 = ConvBlock(filters[3], dropout, kSize, dtype)
        self.convBlock5 = ConvBlock(filters[4], dropout, kSize, dtype)


    def call(self, x):

        e1 = self.convBlock1(x, self.normalize)
        p1 = MaxPooling2D(pool_size=(2,2))(e1)
        e2 = self.convBlock2(p1, self.normalize)
        p2 = MaxPooling2D(pool_size=(2,2))(e2)
        e3 = self.convBlock3(p2, self.normalize)
        p3 = MaxPooling2D(pool_size=(2,2))(e3)
        e4 = self.convBlock4(p3, self.normalize)
        p4 = MaxPooling2D(pool_size=(2,2))(e4)
        bridge = self.convBlock5(p4, self.normalize)

        return e1,e2,e3,e4, bridge


class Decoder(layers.Layer):
    def __init__(self, filters, dropout, normalize, kSize, dtype):
        super(Decoder, self).__init__()
        self.normalize = normalize
        self.convblock1 = ConvBlock(filters[3], dropout, kSize, dtype)
        self.convblock2 = ConvBlock(filters[2], dropout, kSize, dtype)
        self.convblock3 = ConvBlock(filters[1], dropout, kSize, dtype)
        self.convblock4 = ConvBlock(filters[0], dropout, kSize, dtype)

        self.gridGating = GridGatingSignal(filters[3])

        self.attention2 = AttentionBlock(filters[1], filters[1])
        self.attention3 = AttentionBlock(filters[2], filters[2])
        self.attention4 = AttentionBlock(filters[3], filters[3])


    def call(self,e1,e2,e3,e4,bridge):

        gating = self.gridGating(bridge, self.normalize)

        a4 = self.attention4(e4, gating, self.normalize)
        a3 = self.attention3(e3, gating, self.normalize)
        a2 = self.attention2(e2, gating, self.normalize)

        d4 = UpSampling2D((2, 2))(bridge)
        d4 = Concatenate()([d4, a4])
        d4 = self.convblock1(d4, self.normalize)

        d3 = UpSampling2D((2, 2))(d4)
        d3 = Concatenate()([d3, a3])
        d3 = self.convblock2(d3, self.normalize)

        d2 = UpSampling2D((2, 2))(d3)
        d2 = Concatenate()([d2, a2])
        d2 = self.convblock3(d2, self.normalize)

        d1 = UpSampling2D((2, 2))(d2)
        d1 = Concatenate()([d1, e1])
        d1 = self.convblock4(d1, self.normalize)

        return d1


class AttenUnetSC(Model):
    def __init__(self, filters=[16,32,64,128,256], finalActivation='sigmoid', activation='relu', kSize=(3,3),
                 nOutput=1, padding='same', dropout=0, dilation=(1,1), normalize=True, dtype='float32'):
        super(AttenUnetSC, self).__init__()
        self.encoder = Encoder(filters, dropout, normalize, kSize, dtype)
        self.decoder = Decoder(filters, dropout, normalize, kSize, dtype)
        self.finalconv = Conv2D(nOutput, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')

    def call(self, x):

        e1,e2,e3,e4,bridge = self.encoder(x)
        x = self.decoder(e1,e2,e3,e4,bridge)
        x = self.finalconv(x)

        return x


#################################### functional model ####################################


class AttenUnetFunc():
    def __init__(self, filters=[16,32,64,128,256], finalActivation='sigmoid', activation='relu', kSize=(3,3),
                 nOutput=1, padding='same', dropout=0, dilation=(1,1), normalize=True):
        self.filters = filters
        self.activation = activation
        self.finalActivation = finalActivation
        self.padding = padding
        self.normalize = normalize
        self.kernelSize = kSize
        self.nOutput = nOutput
        self.dropout = dropout


    def convBlock(self, x, f, contraction=True):

        x = Conv2D(filters=f, kernel_size=self.kernelSize, activation=self.activation, padding=self.padding)(x)
        x = BatchNormalization()(x) if self.normalize else x
        x = Dropout(self.dropout)(x) if contraction else x
        x = Conv2D(filters=f, kernel_size=self.kernelSize, activation=self.activation, padding=self.padding)(x)
        x = BatchNormalization()(x) if self.normalize else x
        x = Dropout(self.dropout)(x) if contraction else x

        return x


    def attention(self, x, g, f1, f2):

        thetaX = Conv2D(f1, kernel_size=(2,2), strides=(2,2), padding='same', use_bias=False)(x)
        phiG = Conv2D(f1, kernel_size=(1,1),strides=1, padding='same', use_bias=True)(g)
        upFactor = K.int_shape(thetaX)[1]/K.int_shape(phiG)[1]
        phiG = UpSampling2D(size=(upFactor, upFactor), interpolation='bilinear')(phiG)
        psi = Conv2D(1, kernel_size=(1,1), strides=1, padding='same', use_bias=True)(Add()([phiG, thetaX]))
        psi = Activation(activation='relu')(psi)
        psi = Activation(activation='sigmoid')(psi)
        upFactor = K.int_shape(x)[1]/K.int_shape(psi)[1]
        psi = UpSampling2D(size=(upFactor, upFactor), interpolation='bilinear')(psi)
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

        a4 = self.attention(e4, gating, self.filters[3], self.filters[3])
        a3 = self.attention(e3, gating, self.filters[2], self.filters[2])
        a2 = self.attention(e2, gating, self.filters[1], self.filters[1])

        d4 = UpSampling2D((2, 2))(bridge)
        d4 = Concatenate()([d4, a4])
        d4 = self.convBlock(d4, self.filters[3])

        d3 = UpSampling2D((2, 2))(d4)
        d3 = Concatenate()([d3, a3])
        d3 = self.convBlock(d3, self.filters[2])

        d2 = UpSampling2D((2, 2))(d3)
        d2 = Concatenate()([d2, a2])
        d2 = self.convBlock(d2, self.filters[1])

        d1 = UpSampling2D((2, 2))(d2)
        d1 = Concatenate()([d1, e1])
        d1 = self.convBlock(d1, self.filters[0])

        return d1


    def attenunet(self):

        tensorInput = Input((256, 256, 3))
        e1,e2,e3,e4,bridge = self.encoder(tensorInput)
        d = self.decoder(e1,e2,e3,e4,bridge)
        finalMap = Conv2D(self.nOutput, kernel_size=self.kernelSize, strides=(1, 1), activation=self.finalActivation)(d)
        x = Model(tensorInput, finalMap)

        return x
