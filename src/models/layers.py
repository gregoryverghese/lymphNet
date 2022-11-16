import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2,l1
from tensorflow.keras.layers import Conv2D, UpSampling2D,BatchNormalization,GaussianNoise
from tensorflow.keras.layers import MaxPooling2D, Dropout, Activation, Concatenate
from tensorflow.keras.layers import Add, Multiply, Input, Conv2DTranspose,LeakyReLU, ReLU



class ConvLayer():
    def __init__(self,
                kernel_size,
                padding,
                initializer):

        self.kernel_size=kernel_size
        self.padding=padding
        self.initializer=initializer


    def __call__(self, f, dilation=1):
        return Conv2D(
                filters=f, 
                kernel_size=self.kernel_size,
                padding=self.padding, 
                kernel_initializer=self.initializer,
                dilation_rate=dilation
                )


class UpLayer():
    def __init__(self,
                kernel_size,
                padding,
                initializer,
                activation,
                layer_type):
        
        self.kernel_size=kernel_size
        self.padding=padding
        self.initializer=initializer
        self.activation=activation
        self.layer_type=layer_type


    def __call__(self, f=None, pool=(2,2)):
        if self.layer_type=='upsampling':
            layer=UpSampling2D(pool)
        elif self.layer_type=='transpose':
            layer=Conv2DTranspose(
                            f,
                            kernel_size=self.kernel_size,
                            activation=self.activation, 
                            strides=(2,2), 
                            padding=self.padding
                            )
        return layer


class IdentityLayer():
    def __init__(self, padding, strides):
        self.kernel_size=(1,1)
        self.padding=padding
        self.strides=strides
    

    def __call__(self, x , f):
 
        identity = Conv2D(f, 
                          kernel_size=self.kernel_size,
                          padding=self.padding, 
                          strides=self.strides)(x)
        identity = BatchNormalization()(identity)
        x = Add()([identity, x])
        return x


def conv_block(x,
               f,
               conv_layer,
               normalize=True, 
               drop=False):

    x = conv_layer(f)(x) 
    x = BatchNormalization()(x) if normalize else x
    #x = GaussianNoise(0.3)(x)
    x = ReLU()(x)
    #x = LeakyReLU(0.1)(x)
    #x = Dropout(0.2)(x) if contraction else x
    x = conv_layer(f)(x)
    x = BatchNormalization()(x) if normalize else x
    #x = GaussianNoise(0.3)(x)
    #x = LeakyReLU(0.1)(x)
    x = ReLU()(x)
    #x = LeakyReLU(0.1)(x)
    #x = Dropout(0.2)(x) if contraction else x 
    return x



def residual_block(x1,
               f,
               conv_layer,
               identity_layer,
               strides=(1,1),
               normalize=True,
               drop=False):

    x2 = conv_layer(f)(x) 
    x2 = BatchNormalization()(x) if normalize else x
    x2 = ReLU()(x)
    #x = Dropout(0.2)(x) if drop else x
    x2 = conv_layer(f)(x)
    x2 = BatchNormalization()(x) if normalize else x
    x2 = ReLU()(x)
    #x = Dropout(0.2)(x) if drop else x 
    x2 = identity_layer(r, x, f, strides=stride)
    return x


def multi_block(x,
                f,
                conv_layer,
                up_layer,
                normalize=True
               ):

    out_f=f/3
    x = MaxPooling2D((2,2))(x)

    x1 = conv_layer(out_f, 1)(x)
    x1 = MaxPooling2D((1,1))(x1)
    x1 = BatchNormalization()(x1) if normalize else x1
    x1 = ReLU()(x1)
    x1 = conv_layer(out_f, 1)(x1)
    x1 = BatchNormalization()(x1) if normalize else x1
    x1 = ReLU()(x1)

    x2 = conv_layer(out_f, 2)(x1)
    x2 = MaxPooling2D((2,2))(x2)
    x2 = BatchNormalization()(x2) if normalize else x2
    x2 = ReLU()(x2)
    x2 = conv_layer(out_f, 2)(x2)
    x2 = BatchNormalization()(x2) if normalize else x2
    x2 = ReLU()(x2)
    x2 = up_layer(pool=(2,2))(x2)

    x3 = conv_layer(out_f, 4)(x2)
    x3 = MaxPooling2D((4,4))(x3)
    x3 = BatchNormalization()(x3) if normalize else x3
    x3 = ReLU()(x3)
    x3 = conv_layer(out_f, 4)(x3)
    x3 = BatchNormalization()(x3) if normalize else x3
    x3 = ReLU()(x3) 
    x3 = up_layer(pool=(4,4))(x3)

    x = Concatenate()([x1,x2,x3])

    return x






