
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2,l1
from tensorflow.keras.layers import Conv2D, UpSampling2D,BatchNormalization,GaussianNoise
from tensorflow.keras.layers import MaxPooling2D, Dropout, Activation, Concatenate
from tensorflow.keras.layers import Add, Multiply, Input, Conv2DTranspose,LeakyReLU, ReLU
#from crfrnn_layer import CrfRnnLayer




class ConvBlock(layers.Layer):
    def __init__(self, f, dropout, kSize, dtype):
        super(ConvBlock, self).__init__(dtype=dtype)
        self.f = f
        self.batchnorm1 = BatchNormalization()
        self.batchnorm2 = BatchNormalization()
        self.drop1 = Dropout(dropout)
        self.drop2 = Dropout(dropout)
        self.conv2d1 = Conv2D(f, kSize, activation='relu', padding='same')
        self.conv2d2 = Conv2D(f, kSize, activation='relu', padding='same')


    def call(self, x, normalize=True):

        x = self.conv2d1(x)
        x = self.batchnorm1(x)
        x = self.drop1(x)
        x = self.conv2d2(x)
        x = self.batchnorm2(x)
        x = self.drop2(x)

        return x


class UnetSC(Model):
    def __init__(self, filters=[16,32,64,128, 256], finalActivation='sigmoid', activation='relu',
                    nOutput=1, kSize=(3,3), pSize=(2,2), dropout=0,
                 normalize=True, padding='same', upTypeName='upsampling', dtype='float32'):
        super(UnetSC, self).__init__(dtype=dtype)

        self.normalize = normalize
        self.convblocke1 = ConvBlock(filters[0], dropout, kSize, dtype)
        self.pool1 = MaxPooling2D((2, 2))
        self.convblocke2 = ConvBlock(filters[1], dropout, kSize, dtype)
        self.pool2 = MaxPooling2D((2, 2))
        self.convblocke3 = ConvBlock(filters[2], dropout, kSize, dtype)
        self.pool3 = MaxPooling2D((2, 2))
        self.convblocke4 = ConvBlock(filters[3], dropout, kSize, dtype)
        self.pool4 = MaxPooling2D((2, 2))

        self.convb_1 = Conv2D(filters[4], kSize, activation='relu', padding='same')
        self.batchnorm9 = BatchNormalization()
        self.convb_2 = Conv2D(filters[4], kSize, activation='relu', padding='same')
        self.batchnorm10 = BatchNormalization()

        if upTypeName=='upsampling':
            self.up1 = UpSampling2D((2, 2))
        elif upTypeName=='transpose':
            self.up1 = Conv2DTranspose(filters[4], kSize, activation='relu', strides=(2,2), padding='same')
        self.conc1 = Concatenate()
        self.convblockd1 = ConvBlock(filters[3], dropout, kSize, dtype)

        if upTypeName=='upsampling':
            self.up2 = UpSampling2D((2, 2))
        elif upTypeName=='transpose':
            self.up2 = Conv2DTranspose(filters[3], kSize, activation='relu',strides=(2,2), padding='same')

        self.conc2 = Concatenate()
        self.convblockd2 = ConvBlock(filters[2], dropout, kSize, dtype)

        if upTypeName=='upsampling':
            self.up3 = UpSampling2D((2, 2))
        elif upTypeName=='transpose':
            self.up3 = Conv2DTranspose(filters[2], kSize, activation='relu',strides=(2,2),padding='same')
        self.conc3 = Concatenate()
        self.convblockd3 = ConvBlock(filters[1], dropout, kSize, dtype)

        if upTypeName=='upsampling':
            self.up4 = UpSampling2D((2, 2))
        elif upTypeName=='transpose':
            self.up4 = Conv2DTranspose(filters[1], kSize, activation='relu',strides=(2,2), padding='same')

        self.conc4 = Concatenate()
        self.convblockd4 = ConvBlock(filters[0], dropout, kSize, dtype)

        self.final = Conv2D(nOutput, kernel_size=(1, 1), strides=(1, 1), activation=finalActivation)


    def call(self, x, training=True):

        e1 = self.convblocke1(x)
        p1 = self.pool1(e1)

        e2 = self.convblocke2(p1)
        p2 = self.pool2(e2)

        e3 = self.convblocke3(p2)
        p3 = self.pool3(e3)

        e4 = self.convblocke4(p3)
        p4 = self.pool4(e4)

        b = self.convb_1(p4)
        b = self.batchnorm9(b)
        b = self.convb_2(b)
        b = self.batchnorm10(b)

        d1 = self.up1(b)
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


        x = self.final(d4)

        return x
