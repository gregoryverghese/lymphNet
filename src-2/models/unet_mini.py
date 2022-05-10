import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model


class UnetMini():
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
