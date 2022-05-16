import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, MaxPooling2D, Dropout, Activation, Concatenate, Add, Multiply
from tensorflow.keras import layers
from tensorflow.keras import Model

#NOT REFACTORED

class ResUnet():
    def __init__(self, 
                 filters=[32,64,128,256,512], 
                 final_activation='sigmoid', 
                 activation='relu', 
                 kernel_size=(3,3),
                 intializer='glorot_uniform',
                 n_output=1, 
                 dropout=0, 
                 normalize=True, 
                 padding='same'):

        self.filters = filters
        self.activation = activation
        self.final_activation = final_activation
        self.kernel_size=kernel_size
        self.initializer = initializer
        self.padding = padding
        self.n_output = n_output
        self.dropout = dropout


    @property
    def conv_layer(self):
        return ConvLayer(
             self.kernel_size,
             self.padding,
             self.initializer
             )


    @property
    def identity_layer(self):
        return IdentityLayer(
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


    def residualBlock(self, xIn, f, stride):

        res = self.convBlocks(xIn, f, strides=stride)
        res = keras.layers.Dropout(self.dropout)(res)
        res = self.convBlocks(res, f, strides=1)
        res = keras.layers.Dropout(self.dropout)(res)
        output = self.identity(res, xIn, f, strides=stride)

        return output


    def upSampling(self, x, xInput):

        x = keras.layers.UpSampling2D((2,2))(x)
        x = keras.layers.Concatenate()([x, xInput])

        return x


    def encoder(self, x, kernelSize=(3,3), padding='same', strides=1, dropout=0):

        e1 = keras.layers.Conv2D(self.filters[0], kernelSize, padding=padding, strides=strides)(x)
        e1 = self.convBlocks(e1, self.filters[0])
        e1 = keras.layers.Dropout(self.dropout)(e1)

        shortcut = keras.layers.Conv2D(self.filters[0], kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = keras.layers.BatchNormalization()(shortcut)
        e1Output = keras.layers.Add()([e1, shortcut])

        e2 = self.residualBlock(e1Output, self.filters[1], stride=2)
        e3 = self.residualBlock(e2, self.filters[2], stride=2)
        e4 = self.residualBlock(e3, self.filters[3], stride=2)
        e5 = self.residualBlock(e4, self.filters[4], stride=2)

        return e1Output, e2, e3, e4, e5


    def bridge(self, x, f):

        b1 = self.convBlocks(x, f, strides=1)
        b2 = self.convBlocks(b1, f, strides=1)

        return b2


    def decoder(self, b2, e1, e2, e3, e4, kernelSize=(3,3), padding='same', strides=1):

        x = self.upSampling(b2, e4)
        d1 = self.convBlocks(x, self.filters[4])
        d1 = self.convBlocks(d1, self.filters[4])
        d1 = self.identity(d1, x, self.filters[4])

        x = self.upSampling(d1, e3)
        d2 = self.convBlocks(x, self.filters[3])
        d2 = self.convBlocks(d2, self.filters[3])
        d2 = self.identity(d2, x, self.filters[3])

        x = self.upSampling(d2, e2)
        d3 = self.convBlocks(x, self.filters[2])
        d3 = self.convBlocks(d3, self.filters[2])
        d3 = self.identity(d3, x, self.filters[2])

        x = self.upSampling(d3, e1)
        d4 = self.convBlocks(x, self.filters[1])
        d4 = self.convBlocks(d4, self.filters[1])
        d4 = self.identity(d4, x, self.filters[1])

        return d4


    def build(self):

        inputs = keras.layers.Input((None, None, 3))

        e1, e2, e3, e4, e5 = self.encoder(inputs)
        b2 = self.bridge(e5, self.filters[4])
        d4 = self.decoder(b2, e1, e2, e3, e4)

        x = keras.layers.Conv2D(self.nOutput, (1, 1), padding='same', activation=self.finalActivation)(d4)
        model = keras.models.Model(inputs, x)

        return model
