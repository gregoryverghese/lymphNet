import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Dropout, Input, concatenate


class Unet():

    def __init__(self, imgDims, activation='relu', finalActivation='sigmoid', nOutput=1, dropout=0, normalize=True, padding='same'):
        self.imgDims = imgDims
        self.activaion = activation
        self.finalActivation = finalActivation
        self.nOutput = nOutput
        self.dropout=dropout
        self.normalize = normalize
        self.padding = padding


    def convBlockContraction(self, x, f, activation='relu', kSize=(3, 3), pSize=(2, 2)):

        x = Conv2D(filters=f, kernel_size=kSize, activation='relu', padding=self.padding)(x)

        if self.normalize:
            x = BatchNormalization()(x)

        x = Dropout(self.dropout)(x)
        x = Conv2D(filters=f, kernel_size=kSize, activation='relu', padding=self.padding)(x)

        if self.normalize:
            x = BatchNormalization()(x)

        p = MaxPooling2D(pool_size=pSize)(x)

        return p, x


    def convBlockExpansion(self, x, x2, f, kSize=(3, 3)):

        x = UpSampling2D((2, 2))(x)
        x = Conv2D(f, (2, 2), activation='relu', padding=self.padding)(x)
        if self.normalize:
            x = BatchNormalization()(x)

        if self.padding=='valid':
            x2 = Cropping2D((5, 5))(x2)

        x = concatenate([x2, x])

        x = Conv2D(f, kSize, activation='relu', padding=self.padding)(x)
        if self.normalize:
            x = BatchNormalization()(x)

        x = Conv2D(f, kSize, activation='relu', padding=self.padding)(x)
        if self.normalize:
            x = BatchNormalization()(x)

        return x


    def bridge(self, x, f, kSize=(3, 3)):

        x = Conv2D(f, kSize, padding=self.padding)(x)
        if self.normalize:
            x = BatchNormalization()(x)
        x = Conv2D(f, kSize, padding=self.padding)(x)
        if self.normalize:
            x = BatchNormalization()(x)

        return x


    def encoder(self, x):

        e1, c1 = self.convBlockContraction(x, 8)
        e2, c2 = self.convBlockContraction(e1, 16)
        e3, c3 = self.convBlockContraction(e2, 32)
        e4, c4 = self.convBlockContraction(e3, 64)
        e5 = self.bridge(e4, 128)

        return c1, c2, c3, c4, e5


    def decoder(self, maps):

        c1, c2, c3, c4, e5 = maps
        d1 = self.convBlockExpansion(e5, c4, 64, kSize=(3, 3))
        d2 = self.convBlockExpansion(d1, c3, 32, kSize=(3, 3))
        d3 = self.convBlockExpansion(d2, c2, 16, kSize=(3, 3))
        d4 = self.convBlockExpansion(d3, c1, 8, kSize=(3, 3))

        return d4



    def unet(self):

        tensorInput = Input((None, None, 3))
        encodeMaps = self.encoder(tensorInput)
        decodeMap = self.decoder(encodeMaps)
        finalMap = Conv2D(self.nOutput, (1, 1), activation=self.finalActivation)(decodeMap)
        model = Model(tensorInput, finalMap)

        return model
