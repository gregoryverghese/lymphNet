import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, MaxPooling2D, Dropout, Activation, Concatenate, Add, Multiply, Input, Conv2DTranspose

#################################### subclassing model ####################################

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
            self.up1 = Conv2DTranspose(filters[4], kSize, activation='relu', stride=(2,2), padding='same')

        self.conc1 = Concatenate()
        self.convblockd1 = ConvBlock(filters[3], dropout, kSize, dtype)

        if upTypeName=='upsampling':
            self.up2 = UpSampling2D((2, 2))
        elif upTypeName=='transpose':
            self.up2 = Conv2DTranspose(filters[4], kSize, activation='relu', stride=(2,2), padding='same')

        self.conc2 = Concatenate()
        self.convblockd2 = ConvBlock(filters[2], dropout, kSize, dtype)

        if upTypeName=='upsampling':
            self.up3 = UpSampling2D((2, 2))
        elif upTypeName=='transpose':
            self.up3 = Conv2DTranspose(filters[2], kSize, activation='relu', stride=(2,2),padding='same')

        self.conc3 = Concatenate()
        self.convblockd3 = ConvBlock(filters[1], dropout, kSize, dtype)

        if upTypeName=='upsampling':
            self.up4 = UpSampling2D((2, 2))
        elif upTypeName=='transpose':
            self.up4 = Conv2DTranspose(filters[4], kSize, activation='relu', stride=(2,2), padding='same')

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

        d1 = self.upsampling1(b)
        d1 = self.conc1([e4, d1])
        d1 = self.convblockd1(d1)

        d2 = self.upsampling2(d1)
        d2 = self.conc2([e3, d2])
        d2 = self.convblockd2(d2)

        d3 = self.upsampling3(d2)
        d3 = self.conc3([e2, d3])
        d3 = self.convblockd3(d3)


        d4 = self.upsampling4(d3)
        d4 = self.conc4([e1, d4])
        d4 = self.convblockd4(d4)


        x = self.final(d4)

        return x

#################################### functional model ####################################

class UnetFunc():
    def __init__(self, filters=[32,64,128,256,512], finalActivation='sigmoid', activation='relu',
                nOutput=1, kernelSize=(3,3), pSize=(2,2), dropout=0,
                 normalize=True, padding='same', upTypeName='upsampling', dtype='float32'):

        self.filters = filters
        self.activation = activation
        self.finalActivation = finalActivation
        self.nOutput = nOutput
        self.kernelSize = kernelSize
        self.pSize = pSize
        self.dropout=dropout
        self.normalize = normalize
        self.padding = padding
        self.upTypeName = upTypeName
        self.dtype = dtype


    def convBlock(self, x, f, contraction=True):

        x = Conv2D(filters=f, kernel_size=self.kernelSize, activation=self.activation, padding=self.padding)(x)
        x = BatchNormalization()(x) if self.normalize else x
        x = Dropout(self.dropout)(x) if contraction else x
        x = Conv2D(filters=f, kernel_size=self.kernelSize, activation=self.activation, padding=self.padding)(x)
        x = BatchNormalization()(x) if self.normalize else x
        x = Dropout(self.dropout)(x) if contraction else x

        return x


    def bridge(self, x, f, kSize=(3, 3)):

        x = Conv2D(f, kSize, padding=self.padding)(x)
        x = BatchNormalization()(x) if self.normalize else x
        x = Conv2D(f, kSize, padding=self.padding)(x)
        x = BatchNormalization()(x) if self.normalize else x

        return x


    def encoder(self, inputTensor):

        e1 = self.convBlock(inputTensor, self.filters[0])
        p1 = MaxPooling2D((2,2))(e1)
        e2 = self.convBlock(p1, self.filters[1])
        p2 = MaxPooling2D((2,2))(e2)
        e3 = self.convBlock(p2, self.filters[2])
        p3 = MaxPooling2D((2,2))(e3)
        e4 = self.convBlock(p3, self.filters[3])
        p4 = MaxPooling2D((2,2))(e4)
        bridge = self.bridge(p4, self.filters[4])

        return e1,e2,e3,e4,bridge



    def decoder(self, e1,e2,e3,e4, bridge):

        if self.upTypeName=='upsampling':
            d5 = UpSampling2D((2,2))(bridge)
        elif self.upTypeName=='transpose':
            d5 = Conv2DTranspose(self.filters[4],self.kernelSize,activation='relu', strides=(2,2), padding='same')(bridge)

        d5 = Concatenate()([e4, d5])
        d5 = self.convBlock(d5, self.filters[3], contraction=False)

        if self.upTypeName=='upsampling':
            d4 = UpSampling2D((2,2))(d5)
        elif self.upTypeName=='transpose':
            d4 = Conv2DTranspose(self.filters[3], self.kernelSize, activation='relu', strides=(2,2),padding='same')(d5)

        d4 = Concatenate()([e3, d4])
        d4 = self.convBlock(d4, self.filters[2], contraction=False)
        
        if self.upTypeName=='upsampling':
            d3 = UpSampling2D((2,2))(d4)
        elif self.upTypeName=='transpose':
            d3 = Conv2DTranspose(self.filters[2], self.kernelSize, activation='relu', strides=(2,2), padding='same')(d4)

        d3 = Concatenate()([e2, d3])
        d3 = self.convBlock(d3, self.filters[1], contraction=False)

        if self.upTypeName=='upsampling':
            d2 = UpSampling2D((2,2))(d3)
        elif self.upTypeName=='transpose':
            d2 = Conv2DTranspose(self.filters[1], self.kernelSize,activation='relu', strides=(2,2), padding='same')(d3)

        d2 = Concatenate()([e1, d2])
        d2 = self.convBlock(d2, self.filters[0], contraction=False)

        return d2


    def unet(self):

        tensorInput = Input((None, None, 3))
        e1,e2,e3,e4,bridge = self.encoder(tensorInput)
        d2 = self.decoder(e1,e2,e3,e4, bridge)
        finalMap = Conv2D(self.nOutput, (1, 1), strides=(1,1), activation=self.finalActivation)(d2)
        model = Model(tensorInput, finalMap)

        return model
