import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, MaxPooling2D, Dropout, Activation, Concatenate, Add, Multiply

#################################### subclassing model ####################################

class ConvBlock(layers.Layer):
    def __init__(self, f, dropout, kSize, dtype):
        super(ConvBlock, self).__init__(dtype=dtype)
        self.f = f

        self.batchnorm1 = BatchNormalization()
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.batchnorm2 = BatchNormalization()
        self.conv2d1 = Conv2D(f, kSize, activation='relu', padding='same')
        self.conv2d2 = Conv2D(f, kSize, activation='relu', padding='same')


    def call(self, x, normalize, training):
        
        x = self.conv2d1(x)

        if training:
           x = self.batchnorm1(x, training=training) if normalize else x
           x = self.dropout1(x, training=training)

        x = self.conv2d2(x)

        if training:
           x = self.batchnorm2(x, training) if normalize else x
           x = self.dropout2(x, training=training)

        return x


class Bridge(layers.Layer):
    def __init__(self, f, kSize, dtype):
        super(Bridge, self).__init__(dtype)
        self.conv2d1 = Conv2D(f, kSize, activation='relu', padding='same')
        self.batchnorm1 = BatchNormalization()
        self.batchnorm2 = BatchNormalization()
        self.conv2d2 = Conv2D(f, kSize, activation='relu', padding='same')


    def call(self, x, training, normalize=True):

        x = self.conv2d1(x)
        if training:
            x = self.batchnorm1(x, training=training) if normalize else x
        
        x = self.conv2d2(x)
        if training:
            x = self.batchnorm2(x, training=training) if normalize else x

        return x


class Encoder(layers.Layer):
    def __init__(self, filters, dropout, normalize, kSize, dtype):
        super(Encoder, self).__init__()
        self.normalize = normalize
        self.convblock1 = ConvBlock(filters[0], dropout, kSize, dtype)
        self.convblock2 = ConvBlock(filters[1], dropout, kSize, dtype)
        self.convblock3 = ConvBlock(filters[2], dropout, kSize, dtype)
        self.convblock4 = ConvBlock(filters[3], dropout, kSize, dtype)
        self.bridge = Bridge(filters[4], kSize, dtype)
        self.pooling1 = MaxPooling2D((2, 2))
        self.pooling2 = MaxPooling2D((2, 2))
        self.pooling3 = MaxPooling2D((2, 2))
        self.pooling4 = MaxPooling2D((2, 2))


    def call(self, inputTensor, training):
        e1 = self.convblock1(inputTensor, self.normalize, training)
        p1 = self.pooling1(e1)
        e2 = self.convblock2(p1, self.normalize, training)
        p2 = self.pooling2(e2)
        e3 = self.convblock3(p2, self.normalize, training)
        p3 = self.pooling3(e3)
        e4 = self.convblock4(p3, self.normalize, training)
        p4 = self.pooling4(e4)
        bridge = self.bridge(p4, self.normalize, training)

        return e1,e2,e3,e4,bridge


class Decoder(layers.Layer):
    def __init__(self, filters, dropout, normalize, kSize, dtype):
        super(Decoder, self).__init__()
        self.normalize = normalize
        self.upsampling1 = UpSampling2D((2, 2))
        self.upsampling2 = UpSampling2D((2, 2))
        self.upsampling3 = UpSampling2D((2, 2))
        self.upsampling4 = UpSampling2D((2, 2))
        self.convblock1 = ConvBlock(filters[3], dropout, kSize, dtype)
        self.convblock2 = ConvBlock(filters[2], dropout, kSize, dtype)
        self.convblock3 = ConvBlock(filters[1], dropout, kSize, dtype)
        self.convblock4 = ConvBlock(filters[0], dropout, kSize, dtype)


    def call(self, e1,e2,e3,e4, bridge, training):
        d5 = self.upsampling1(bridge)
        d5 = Concatenate()([e4, d5])
        d5 = self.convblock1(d5, self.normalize, training)
        d4 = self.upsampling2(d5)
        d4 = Concatenate()([e3, d4])
        d4 = self.convblock2(d4, self.normalize, training)
        d3 = self.upsampling3(d4)
        d3 = Concatenate()([e2, d3])
        d3 = self.convblock3(d3, self.normalize, training)
        d2 = self.upsampling4(d3)
        d2 = Concatenate()([e1, d2])
        d2 = self.convblock4(d2, self.normalize, training)

        return d2


class UnetSC(Model):
    def __init__(self, filters=[32,64,128,256,512], finalActivation='sigmoid', activation='relu',
                    nOutput=1, kSize=(3,3), pSize=(2,2), dropout=0, normalize=True, padding='same', dtype='float32'):
        super(UnetSC, self).__init__(dtype=dtype)
        self.encoder = Encoder(filters, dropout, normalize, kSize, dtype)
        self.decoder = Decoder(filters, dropout, normalize, kSize, dtype)
        self.final = Conv2D(nOutput, kernel_size=(1, 1), strides=(1, 1), activation=finalActivation)


    def call(self, x, training=True):
        e1,e2,e3,e4,bridge = self.encoder(x, training)
        x = self.decoder(e1,e2,e3,e4,bridge, training)
        x = self.final(x)

        return x

#################################### functional model ####################################

class UnetFunc():
    def __init__(self, filters=[32,64,128,256,512], finalActivation='sigmoid', activation='relu',
                nOutput=1, kernelSize=(3,3), pSize=(2,2), dropout=0, normalize=True, padding='same', dtype='float32'):

        self.filters = filters
        self.activation = activation
        self.finalActivation = finalActivation
        self.nOutput = nOutput
        self.kernelSize = kernelSize
        self.pSize = pSize
        self.dropout=dropout
        self.normalize = normalize
        self.padding = padding
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

        d5 = UpSampling2D((2,2))(bridge)
        d5 = Concatenate()([e4, d5])
        d5 = self.convBlock(d5, self.filters[3], contraction=False)
        d4 = UpSampling2D((2,2))(d5)
        d4 = Concatenate()([e3, d4])
        d4 = self.convBlock(d4, self.filters[2], contraction=False)
        d3 = UpSampling2D((2,2))(d4)
        d3 = Concatenate()([e2, d3])
        d3 = self.convBlock(d3, self.filters[1], contraction=False)
        d2 = UpSampling2D((2,2))(d3)
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
