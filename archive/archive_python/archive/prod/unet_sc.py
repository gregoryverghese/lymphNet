import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import Conv2D, Upsampling2D, BatchNormalization, MaxPooling2D, Dropout, Activation, Concatenate, Add, Multiply


class ConvBlock(layers.Layer):
    def __init__(self, f, dtype='float16', kSize=(3,3), drop=0):
        super(ConvBlock, self).__init__(dtype=dtype)
        self.f = f
        self.batchnorm = BatchNormalization()
        self.dropout = Dropout(0)
        self.conv2d1 = Conv2D(f, kSize, activation='relu', padding='same')
        self.conv2d2 = Conv2D(f, kSize, activation='relu', padding='same')


    def call(self, x):
        x = self.conv2d1(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.conv2d2(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        return x


class Bridge(layers.Layer):
    def __init__(self, f):
        super(Bridge, self).__init__()
        self.conv2d1 = Conv2D(f, (3, 3), activation='relu', padding='same')
        self.batchnorm = BatchNormalization()
        self.conv2d2 = Conv2D(f, (3, 3), activation='relu', padding='same')

    def call(self, x):
        x = self.conv2d1(x)
        x = self.batchnorm(x)
        x = self.conv2d2(x)
        x = self.batchnorm(x)
        return x

class Encoder(layers.Layer):
    def __init__(self, filters):
        super(Encoder, self).__init__()
        self.convblock1 = ConvBlock(filters[0])
        self.convblock2 = ConvBlock(filters[1])
        self.convblock3 = ConvBlock(filters[2])
        self.convblock4 = ConvBlock(filters[3])
        self.pooling = MaxPooling2D((2, 2))
        self.bridge = Bridge(filters[4])


    def call(self, inputTensor):
        e1 = self.convblock1(inputTensor)
        p1 = self.pooling(e1)
        e2 = self.convblock2(p1)
        p2 = self.pooling(e2)
        e3 = self.convblock3(p2)
        p3 = self.pooling(e3)
        e4 = self.convblock4(p3)
        p4 = self.pooling(e4)
        bridge = self.bridge(p4)

        return e1,e2,e3,e4,bridge


class Decoder(layers.Layer):
    def __init__(self, filters):
        super(Decoder, self).__init__()
        self.upsampling = UpSampling2D((2, 2))
        self.convblock1 = ConvBlock(filters[3])
        self.convblock2 = ConvBlock(filters[2])
        self.convblock3 = ConvBlock(filters[1])
        self.convblock4 = ConvBlock(filters[0])

    def call(self, e1,e2,e3,e4, bridge):
        d5 = self.upsampling(bridge)
        d5 = Concatenate()([e4, d5])
        d5 = self.convblock1(d5)
        d4 = self.upsampling(d5)
        d4 = Concatenate()([e3, d4])
        d4 = self.convblock2(d4)
        d3 = self.upsampling(d4)
        d3 = Concatenate()([e2, d3])
        d3 = self.convblock3(d3)
        d2 = self.upsampling(d3)
        d2 = Concatenate()([e1, d2])
        d2 = self.convblock4(d2)

        return d2


class Unet(Model):
    def __init__(self, filters, dtype, nOutput, finalActivation):
        super(Unet, self).__init__(dtype=dtype)
        self.encoder = Encoder(filters)
        self.decoder = Decoder(filters)
        self.final = Conv2D(nOutput, kernel_size=(1, 1), strides=(1, 1), activation=finalActivation)

    def call(self, x):
        e1,e2,e3,e4,bridge = self.encoder(x)
        x = self.decoder(e1,e2,e3,e4,bridge)
        x = self.final(x)

        return x
