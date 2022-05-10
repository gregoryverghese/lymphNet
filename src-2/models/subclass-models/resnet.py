
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, MaxPooling2D, Dropout, Activation, Concatenate, Add, Multiply
from tensorflow.keras import layers
from tensorflow.keras import Model

#################################### subclassing model ####################################

class Identity(layers.Layer):
    def __init__(self, f, strides=1):
        super(Identity, self).__init__()
        self.conv2D = Conv2D(f, kernel_size=(1, 1), strides=strides, padding='same')
        self.batchnorm = BatchNormalization()

    def call(self, x):
        x = self.conv2D(x)
        x = self.batchnorm(x)
        return x


class ResidualBlock(layers.Layer):
    def __init__(self, f, kSize=(3, 3), padding='same', strides=1, activation='relu'):
        super(ResidualBlock, self).__init__()

        self.batchnorm1 = BatchNormalization()
        self.activation1 = Activation(activation=activation)
        self.conv2d1 = Conv2D(f, kernel_size=kSize, padding=padding, strides=strides, activation='relu')
        self.batchnorm2 = BatchNormalization()
        self.activation2 = Activation(activation=activation)
        self.conv2d2 = Conv2D(f, kernel_size=kSize, padding=padding, strides=1, activation='relu')
        self.identity = Identity(f, strides)

    def call(self, x, stem=False):

        if stem==False:
            x2 = self.batchnorm1(x)
            x2 = self.activation1(x2)
            x2 = self.conv2d1(x2)
        else:
            x2 = self.conv2d1(x)

        x2 = self.batchnorm2(x2)
        x2 = self.activation2(x2)
        x2 = self.conv2d2(x2)
        x  = self.identity(x)
        x2 = Add()([x, x2])

        return x2


class Bridge(layers.Layer):
    def __init__(self, f, activation='relu'):
        super(Bridge, self).__init__()
        self.batchnorm1 = BatchNormalization()
        self.activation1 = Activation(activation=activation)
        self.conv1 = Conv2D(f, kernel_size=(3,3), strides=1, padding='same')
        self.batchnorm2 = BatchNormalization()
        self.activation2 = Activation(activation=activation)
        self.conv2 = Conv2D(f, kernel_size=(3,3), strides=1, padding='same')

    def call(self, x):


        x = self.batchnorm1(x)
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        x = self.conv2(x)
        return x



class ResUnetSC(Model):
    def __init__(self, filters=[32,64,128,256,512], finalActivation='sigmoid', activation='relu', nOutput=1, dropout=0, normalize=True, padding='same'):
        super(ResUnetSC, self).__init__()

        self.residualblock1 = ResidualBlock(filters[0], strides=1)
        self.residualblock2 = ResidualBlock(filters[1], strides=2)
        self.residualblock3 = ResidualBlock(filters[2], strides=2)
        self.residualblock4 = ResidualBlock(filters[3], strides=2)
        self.residualblock5 = ResidualBlock(filters[4], strides=2)

        self.bridge = Bridge(filters[4])

        self.upsampling1 = UpSampling2D((2, 2))
        self.conc1 = Concatenate()
        self.residualblockd1 = ResidualBlock(filters[4])
        self.upsampling2 = UpSampling2D((2, 2))
        self.conc2 = Concatenate()
        self.residualblockd2 = ResidualBlock(filters[3])
        self.upsampling3 = UpSampling2D((2, 2))
        self.conc3 = Concatenate()
        self.residualblockd3 = ResidualBlock(filters[2])
        self.upsampling4 = UpSampling2D((2, 2))
        self.conc4 = Concatenate()
        self.residualblockd4 = ResidualBlock(filters[1])
        self.conv5 = Conv2D(nOutput, (1, 1), padding='same', activation=finalActivation)


    def call(self, x):

        e1 = self.residualblock1(x, stem=True)
        e2 = self.residualblock2(e1)
        e3 = self.residualblock3(e2)
        e4 = self.residualblock4(e3)
        e5 = self.residualblock5(e4)

        bridge = self.bridge(e5)

        d5 = self.upsampling1(bridge)
        d5 = self.conc1([e4, d5])
        d5 = self.residualblockd1(d5)

        d4 = self.upsampling2(d5)
        d4 = self.conc2([e3, d4])
        d4 = self.residualblockd2(d4)

        d3 = self.upsampling3(d4)
        d3 = self.conc3([e2, d3])
        d3 = self.residualblockd3(d3)

        d2 = self.upsampling4(d3)
        d2 = self.conc4([e1, d2])
        d2 = self.residualblockd4(d2)
        x = self.conv5(d2)

        return x

