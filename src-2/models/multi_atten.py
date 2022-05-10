import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Concatenate,concatenate, Conv2DTranspose, ReLU
from tensorflow.keras.layers import  Activation, Dropout, BatchNormalization, MaxPooling2D, Add, Multiply
from tensorflow.keras.regularizers import l2



class MultiAtten():
    def __init__(self, filters=[32,64,128,256,512],finalActivation='sigmoid',
                 activation='relu',nOutput=1,kSize=(3,3),pSize=(2,2),pool=(1,1),padding='same',
                 stride=1,dilation=1,dropout=0,normalize=True,upTypeName='upsampling',dtype='float32'):

        self.normalize = normalize
        self.filters = filters
        self.finalActivation = finalActivation
        self.padding = padding
        self.dropout = dropout
        self.kSize = kSize
        self.pSize = pSize
        self.nOutput = nOutput
        self.stride = stride
        self.pool = pool
        self.dilation =dilation


    def convBlock(self, x, f, dilation=1, pool=(1,1)):

        x = Conv2D(f, kernel_size=(3,3), strides=self.stride,
                   padding=self.padding,dilation_rate=dilation,kernel_regularizer=l2(0.01))(x)
        x = MaxPooling2D((pool))(x)
        x = BatchNormalization()(x)
        x = GaussianNoise(0.2)(x)
        x = Activation('relu')(x)
        #x = Dropout(0.1)(x)
        x = Conv2D(f, kernel_size=(3,3), strides=self.stride,
                   padding=self.padding,dilation_rate=dilation,kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = GaussianNoise(0.2)(x)
        x = Activation('relu')(x)
        #x = Dropout(0.1)(x)

        return x


    def downBlock(self, x, f):

        outChannel = f/3

        x = MaxPooling2D((2,2))(x)
        d1 = self.convBlock(x, outChannel, dilation=self.dilation, pool=self.pool)

        d2 = self.convBlock(x, outChannel, dilation=2, pool=(2,2))
        d2 = UpSampling2D((2,2), interpolation='bilinear')(d2)

        d3 = self.convBlock(x, outChannel, dilation=4, pool=(4,4))
        d3 = UpSampling2D((4,4), interpolation='bilinear')(d3)

        out = Concatenate()([d1,d2,d3])
        return out


    def attention(self, x, g, f1, f2, u):

        thetaX = Conv2D(f1, kernel_size=(2,2), strides=(2,2), padding=self.padding, use_bias=False)(x)
        phiG = Conv2D(f1, kernel_size=(1,1),strides=1, padding=self.padding, use_bias=True)(g)
        phiG = BatchNormalization()(phiG)
        #upFactor = K.int_shape(thetaX)[1]/K.int_shape(phiG)[1]
        upFactor = u
        phiG = UpSampling2D(size=(int(upFactor), int(upFactor)), interpolation='bilinear')(phiG)
        psi = Conv2D(1, kernel_size=(1,1), strides=1, padding=self.padding, use_bias=True)(Add()([phiG, thetaX]))
        psi = BatchNormalization()(psi)
        psi = Activation('relu')(psi)
        psi = Activation('sigmoid')(psi)
        #upFactor = K.int_shape(x)[1]/K.int_shape(psi)[1]
        upFactor = 2
        psi = UpSampling2D(size=(int(upFactor), int(upFactor)), interpolation='bilinear')(psi)
        psi = Multiply()([x, psi])
        psi = Conv2D(f2, kernel_size=(1,1), strides=(1,1), padding=self.padding)(psi)
        psi = BatchNormalization()(psi)

        return psi


    def gridGatingSignal(self, bridge, f):

        x = Conv2D(f, kernel_size=(1,1),strides=(1,1), padding=self.padding)(bridge)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x


    def encoder(self, x):

        x = self.convBlock(x, self.filters[0])
        e1 = self.downBlock(x, self.filters[1])
        e2 = self.downBlock(e1, self.filters[2])
        e3 = self.downBlock(e2, self.filters[3])
        e4 = self.downBlock(e3, self.filters[4])
        e5 = self.downBlock(e4, self.filters[4])

        return x,e1,e2,e3,e4,e5


    def decoder(self, x, e1,e2,e3,e4,e5):

        gating = self.gridGatingSignal(e5, self.filters[3])

        a4 = self.attention(e4, gating, self.filters[3], self.filters[3],1)
        a3 = self.attention(e3, gating, self.filters[2], self.filters[2],2)
        a2 = self.attention(e2, gating, self.filters[1], self.filters[1],4)
        a1 = self.attention(e1, gating, self.filters[1], self.filters[1],8)

        d1 = upsampling1 = UpSampling2D((2, 2))(e5)
        d1 = Concatenate()([d1, a4])
        d1 = self.convBlock(d1, self.filters[4], dilation=1)

        d2 = UpSampling2D((2, 2))(d1)
        d2 = Concatenate()([d2, a3])
        d2 = self.convBlock(d2, self.filters[3], dilation=1)

        d3 = UpSampling2D((2, 2))(d2)
        d3 = Concatenate()([d3, a2])
        d3 = self.convBlock(d3, self.filters[2], dilation=1)

        d4 = UpSampling2D((2, 2))(d3)
        d4 = Concatenate()([d4, a1])
        d4 = self.convBlock(d4, self.filters[1], dilation=1)

        d5 = upsampling5 = UpSampling2D((2, 2))(d4)
        d5 = Concatenate()([d5, x])
        d5 = self.convBlock(d5, self.filters[0], dilation=1)

        return d5


    def build(self):

        tensorInput = Input((None, None, 3))
        x, e1,e2,e3,e4,e5 = self.encoder(tensorInput)
        x = self.decoder(x, e1,e2,e3,e4,e5)
        final = Conv2D(self.nOutput, kernel_size=(1, 1), strides=1, activation=self.finalActivation)(x)
        model = Model(tensorInput, final)

        return model
