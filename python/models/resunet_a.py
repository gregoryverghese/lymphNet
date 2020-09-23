import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Concatenate, concatenate
from tensorflow.keras.layers import  Activation, Dropout, BatchNormalization, MaxPooling2D, Add, Multiply


#################################### subclassing model ####################################


class ConvBlock(layers.Layer):
    def __init__(self, f, d=1, contraction=True):
        super(ConvBlock, self).__init__()
        self.batchnorm1 = BatchNormalization()
        self.conv1 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', dilation_rate=d, padding='same')
        #self.drop1 = Dropout(dropout)
        self.batchnorm2 = BatchNormalization()
        self.conv2 = Conv2D(filters=f, kernel_size=(3,3), activation='relu', dilation_rate=d, padding='same')
        #self.drop2 = Dropout(dropout)


    def call(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        #x = self.drop1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        #x = self.drop2(x)
        return x



class ResUnetABlock(layers.Layer):
    def __init__(self, f, dilationRates):
        super(ResUnetABlock, self).__init__()
        self.convblocks = [ConvBlock(f, d) for d in dilationRates]
        self.add = Add()

    def call(self, x):

        maps = [c(x) for c in self.convblocks]
        mapfinal = self.add(maps) if len(maps) > 1 else maps[0]


        return mapfinal


class PsPPooling(layers.Layer):
    def __init__(self, f):
        super(PsPPooling, self).__init__()

        self.pool1 = MaxPooling2D(pool_size=(1,1))
        self.pool2 = MaxPooling2D(pool_size=(2,2))
        self.pool3 = MaxPooling2D(pool_size=(4,4))
        self.pool4 = MaxPooling2D(pool_size=(8,8))

        self.conv1 = Conv2D(int(f/4), kernel_size=(1,1), strides=(1,1), padding='same')
        self.conv2 = Conv2D(int(f/4), kernel_size=(1,1), strides=(1,1), padding='same')
        self.conv3 = Conv2D(int(f/4), kernel_size=(1,1), strides=(1,1), padding='same')
        self.conv4 = Conv2D(int(f/4), kernel_size=(1,1), strides=(1,1), padding='same')

        self.up1 = UpSampling2D(size=(1,1))
        self.up2 = UpSampling2D(size=(2,2))
        self.up3 = UpSampling2D(size=(4,4))
        self.up4 = UpSampling2D(size=(8,8))

        self.conc = Concatenate()
        self.conv5 = Conv2D(f, (1,1))


    def call(self, x):

        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        x4 = self.pool4(x)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        x1 = self.up1(x1)
        x2 = self.up2(x2)
        x3 = self.up3(x3)
        x4 = self.up4(x4)

        x = self.conc([x1,x2,x3,x4])
        x = self.conv5(x)

        return x


class ResUnetASC(Model):

    def __init__(self, filters, finalActivation='sigmoid'):
        super(ResUnetASC, self).__init__()

        self.conve1 = Conv2D(filters[0], kernel_size=(1, 1), strides=1, padding='same')
        self.resblocke1 = ResUnetABlock(f=filters[0], dilationRates=[1,3,15,31])
        self.pool1 = Conv2D(filters[1], kernel_size=(1, 1), strides=2, padding='same')
        self.resblocke2 = ResUnetABlock(f=filters[1], dilationRates=[1,3,15,31])
        self.pool2 = Conv2D(filters[2], kernel_size=(1, 1), strides=2, padding='same')
        self.resblocke3 = ResUnetABlock(f=filters[2], dilationRates=[1,3,15])
        self.pool3 = Conv2D(filters[3], kernel_size=(1, 1), strides=2, padding='same')
        self.resblocke4 = ResUnetABlock(f=filters[3], dilationRates=[1,3,15])
        self.pool4 = Conv2D(filters[4], kernel_size=(1, 1), strides=2, padding='same')
        self.resblocke5 = ResUnetABlock(f=filters[4], dilationRates=[1])
        self.pool5 = Conv2D(filters[5], kernel_size=(1, 1), strides=2, padding='same')
        self.resblocke6 = ResUnetABlock(f=filters[5], dilationRates=[1])

        self.psp1 = PsPPooling(filters[5])

        self.up1 = UpSampling2D((2, 2))
        self.conc1 = Concatenate()
        self.resblockd1 = ResUnetABlock(filters[5], dilationRates=[1])

        self.up2 = UpSampling2D((2, 2))
        self.conc2 = Concatenate()
        self.resblockd2 = ResUnetABlock(filters[4], dilationRates=[1,3,15])

        self.up3 = UpSampling2D((2, 2))
        self.conc3 = Concatenate()
        self.resblockd3 = ResUnetABlock(filters[3], dilationRates=[1,3,15])

        self.up4 = UpSampling2D((2, 2))
        self.conc4 = Concatenate()
        self.resblockd4 = ResUnetABlock(filters[2], dilationRates=[1,3,15,31])

        self.up5 = UpSampling2D((2, 2))
        self.conc5 = Concatenate()
        self.resblockd5 = ResUnetABlock(filters[1], dilationRates=[1,3,15,31])
        self.conc6 = Concatenate()

        self.psp2 = PsPPooling(filters[0])

        self.convfinal = Conv2D(1, kernel_size=(1, 1), activation=finalActivation, strides=1, padding='same')


    def call(self, x):

        e1 = self.conve1(x)
        e2 = self.resblocke1(e1)
        p2 = self.pool1(e2)
        e3 = self.resblocke2(p2)
        p3 = self.pool2(e3)
        e4 = self.resblocke3(p3)
        p4 = self.pool3(e4)
        e5 = self.resblocke4(p4)
        p5 = self.pool4(e5)
        e6 = self.resblocke5(p5)
        p6 = self.pool5(e6)
        e7 = self.resblocke6(p6)

        psp1 = self.psp1(e7)

        d1 = self.up1(psp1)
        d1 = self.conc1([d1, e6])
        d1 = self.resblockd1(d1)

        d2 = self.up2(d1)
        d2 = self.conc2([d2, e5])
        d2 = self.resblockd2(d2)

        d3 = self.up3(d2)
        d3 = self.conc3([d3, e4])
        d3 = self.resblockd3(d3)

        d4 = self.up4(d3)
        d4 = self.conc4([d4, e3])
        d4 = self.resblockd4(d4)

        d5 = self.up5(d4)
        d5 = self.conc5([d5, e2])
        d5 = self.resblockd5(d5)
        d5 = self.conc6([d5, e1])

        psp2 = self.psp2(d5)

        x = self.convfinal(psp2)

        return x


#################################### functional model ####################################


class ResUnetAFunc():
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
