import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Concatenate, concatenate
from tensorflow.keras.layers import  Activation, Dropout, BatchNormalization, MaxPooling2D, Add, Multiply


class UnetSC(Model):
    def __init__(self, filters=[16,32,64,128, 256], finalActivation='sigmoid', activation='relu',
                    nOutput=1, kSize=(3,3), pSize=(2,2), dropout=0, normalize=True, padding='same', dtype='float32'):
        super(UnetSC, self).__init__(dtype=dtype)

        self.normalize = normalize
        self.conve1_1 = Conv2D(filters[0], kSize, activation='relu', padding='same', name='greg')
        self.batchnorm1 = BatchNormalization(name='greggggggg')
        self.conve1_2 = Conv2D(filters[0], kSize, activation='relu', padding='same')
        self.batchnorm2 = BatchNormalization()
        self.pool1 = MaxPooling2D((2, 2))

        self.conve2_1 = Conv2D(filters[1], kSize, activation='relu', padding='same')
        self.batchnorm3 = BatchNormalization()
        self.conve2_2 = Conv2D(filters[1], kSize, activation='relu', padding='same')
        self.batchnorm4 = BatchNormalization()
        self.pool2 = MaxPooling2D((2, 2))

        self.conve3_1 = Conv2D(filters[2], kSize, activation='relu', padding='same')
        self.batchnorm5 = BatchNormalization()
        self.conve3_2 = Conv2D(filters[2], kSize, activation='relu', padding='same')
        self.batchnorm6 = BatchNormalization()
        self.pool3 = MaxPooling2D((2, 2))

        self.conve4_1 = Conv2D(filters[3], kSize, activation='relu', padding='same')
        self.batchnorm7 = BatchNormalization()
        self.conve4_2 = Conv2D(filters[3], kSize, activation='relu', padding='same', name='finalencoder')
        self.batchnorm8 = BatchNormalization()
        self.pool4 = MaxPooling2D((2, 2))

        self.convb_1 = Conv2D(filters[4], kSize, activation='relu', padding='same')
        self.batchnorm9 = BatchNormalization()
        self.convb_2 = Conv2D(filters[4], kSize, activation='relu', padding='same')
        self.batchnorm10 = BatchNormalization()

        self.upsampling1 = UpSampling2D((2, 2))
        self.conc1 = Concatenate()
        self.convd1_1 = Conv2D(filters[3], kSize, activation='relu', padding='same')
        self.batchnorm11 = BatchNormalization()
        self.convd1_2 = Conv2D(filters[3], kSize, activation='relu', padding='same')
        self.batchnorm12 = BatchNormalization()

        self.upsampling2 = UpSampling2D((2, 2))
        self.conc2 = Concatenate()
        self.convd2_1 = Conv2D(filters[2], kSize, activation='relu', padding='same')
        self.batchnorm13 = BatchNormalization()
        self.convd2_2 = Conv2D(filters[2], kSize, activation='relu', padding='same')
        self.batchnorm14 = BatchNormalization()

        self.upsampling3 = UpSampling2D((2, 2))
        self.conc3 = Concatenate()
        self.convd3_1 = Conv2D(filters[1], kSize, activation='relu', padding='same')
        self.batchnorm15 = BatchNormalization()
        self.convd3_2 = Conv2D(filters[1], kSize, activation='relu', padding='same')
        self.batchnorm16 = BatchNormalization()

        self.upsampling4 = UpSampling2D((2, 2))
        self.conc4 = Concatenate()
        self.convd4_1 = Conv2D(filters[0], kSize, activation='relu', padding='same')
        self.batchnorm17 = BatchNormalization()
        self.convd4_2 = Conv2D(filters[0], kSize, activation='relu', padding='same')
        self.batchnorm18 = BatchNormalization()

        self.final = Conv2D(nOutput, kernel_size=(1, 1), strides=(1, 1), activation=finalActivation)


    def call(self, x, training=True):

        e1 = self.conve1_1(x)
        e1 = self.batchnorm1(e1)
        e1 = self.conve1_2(e1)
        e1 = self.batchnorm2(e1)
        p1 = self.pool1(e1)

        e2 = self.conve2_1(p1)
        e2 = self.batchnorm3(e2)
        e2 = self.conve2_2(e2)
        e2 = self.batchnorm4(e2)
        p2 = self.pool2(e2)

        e3 = self.conve3_1(p2)
        e3 = self.batchnorm5(e3)
        e3 = self.conve3_2(e3)
        e3 = self.batchnorm6(e3)
        p3 = self.pool3(e3)

        e4 = self.conve4_1(p3)
        e4 = self.batchnorm7(e4)
        e4 = self.conve4_2(e4)
        e4 = self.batchnorm8(e4)
        p4 = self.pool4(e4)

        b = self.convb_1(p4)
        b = self.batchnorm9(b)
        b = self.convb_2(b)
        b = self.batchnorm10(b)

        d1 = self.upsampling1(b)
        d1 = self.conc1([e4, d1])
        d1 = self.convd1_1(d1)
        d1 = self.batchnorm11(d1)
        d1 = self.convd1_2(d1)
        d1 = self.batchnorm12(d1)

        d2 = self.upsampling2(d1)
        d2 = self.conc2([e3, d2])
        d2 = self.convd2_1(d2)
        d2 = self.batchnorm13(d2)
        d2 = self.convd2_2(d2)
        d2 = self.batchnorm14(d2)

        d3 = self.upsampling3(d2)
        d3 = self.conc3([e2, d3])
        d3 = self.convd3_1(d3)
        d3 = self.batchnorm15(d3)
        d3 = self.convd3_2(d3)
        d3 = self.batchnorm16(d3)

        d4 = self.upsampling4(d3)
        d4 = self.conc4([e1, d4])
        d4 = self.convd4_1(d4)
        d4 = self.batchnorm17(d4)
        d4 = self.convd4_2(d4)
        d4 = self.batchnorm18(d4)

        x = self.final(d4)

        return x
