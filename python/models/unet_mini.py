import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model


#################################### subclassing model ####################################


class ConvBlock(layers.Layer):
    def __init__(self, f, dropout, kSize, dtype):
        super(ConvBlock, self).__init__()
        self.f = f
        self.batchnorm = BatchNormalization()
        self.dropout = Dropout(dropout)
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


class Bridge(layers.Layer):
    def __init__(self, f):
        super(Bridge, self).__init__()
        self.conv2d1 = Conv2D(f, (3, 3), activation='relu', padding='same')
        self.batchnorm = BatchNormalization()
        self.conv2d2 = Conv2D(f, (3, 3), activation='relu', padding='same')

    def call(self, x,  normalize):
        x = self.conv2d1(x)
        x = self.batchnorm(x) if normalize else x
        x = self.conv2d2(x)
        x = self.batchnorm(x) if normalize else x

        return x


class Encoder(layers.Layer):
    def __init__(self, filters, dropout, normalize, kSize, dtype):
        super(Encoder, self).__init__()
        self.normalize = normalize
        self.convblock1 = ConvBlock(filters[0], dropout, kSize, dtype)
        self.convblock2 = ConvBlock(filters[1], dropout, kSize, dtype)
        self.convblock3 = ConvBlock(filters[2], dropout, kSize, dtype)
        self.pooling = MaxPooling2D((2, 2))
        self.bridge = Bridge(filters[3])


    def call(self, inputTensor):
        e1 = self.convblock1(inputTensor, self.normalize)
        p1 = self.pooling(e1)
        e2 = self.convblock2(p1, self.normalize)
        p2 = self.pooling(e2)
        e3 = self.convblock3(p2, self.normalize)
        p3 = self.pooling(e3)
        bridge = self.bridge(p3, self.normalize)

        return e1,e2,e3,bridge


class Decoder(layers.Layer):
    def __init__(self, filters, dropout, normalize, kSize, dtype):
        super(Decoder, self).__init__()
        self.normalize = normalize
        self.upsampling = UpSampling2D((2, 2))
        self.convblock1 = ConvBlock(filters[2], dropout, kSize, dtype)
        self.convblock2 = ConvBlock(filters[1], dropout, kSize, dtype)
        self.convblock3 = ConvBlock(filters[0], dropout, kSize, dtype)


    def call(self, e1,e2,e3,bridge):

        d3 = self.upsampling(bridge)
        d3 = Concatenate()([e3, d3])
        d3 = self.convblock1(d3, self.normalize)

        d2 = self.upsampling(d3)
        d2 = Concatenate()([e2, d2])
        d2 = self.convblock2(d2, self.normalize)

        d1 = self.upsampling(d2)
        d1 = Concatenate()([e1, d1])
        d1 = self.convblock3(d1, self.normalize)

        return d1


class UnetMiniSC(Model):
    def __init__(self, filters=[32,64,128, 256], finalActivation='sigmoid', activation='relu',
                    nOutput=1, kSize=(3,3), pSize=(2,2), dropout=0, normalize=True, padding='same', dtype='float32'):
        super(UnetMini, self).__init__()
        self.encoder = Encoder(filters, dropout, normalize, kSize, dtype)
        self.decoder = Decoder(filters, dropout, normalize, kSize, dtype)
        self.final = Conv2D(nOutput, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')

    def call(self, x):
        e1,e2,e3,bridge = self.encoder(x)
        x = self.decoder(e1,e2,e3,bridge)
        x = self.final(x)
        return x


#################################### functional model ####################################


class UnetMiniFunc():
    def __init__(self, filters=[32,64,128,256], finalActivation='sigmoid', activation='relu',
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
        bridge = self.bridge(p3, self.filters[3])

        return e1,e2,e3,bridge


    def decoder(self, e1,e2,e3, bridge):

        d3 = UpSampling2D((2,2))(bridge)
        d3 = Concatenate()([e3, d3])
        d3 = self.convBlock(d3, self.filters[2], contraction=False)
        d2 = UpSampling2D((2,2))(bridge)
        d2 = Concatenate()([e2, d3])
        d2 = self.convBlock(d2, self.filters[1], contraction=False)
        d1 = UpSampling2D((2,2))(bridge)
        d1 = Concatenate()([e1, d2])
        d1 = self.convBlock(d1, self.filters[0], contraction=False)

        return d1



    def build(self):

        tensorInput = Input((None, None, 3))
        e1,e2,e3,bridge = self.encoder(tensorInput)
        d1 = self.decoder(e1,e2,e3,bridge)
        finalMap = Conv2D(self.nOutput, (1, 1), activation=self.finalActivation)(d1)
        model = Model(tensorInput, finalMap)

        return model
