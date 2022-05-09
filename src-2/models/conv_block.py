import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2,l1
from tensorflow.keras.layers import Conv2D, UpSampling2D,BatchNormalization,GaussianNoise
from tensorflow.keras.layers import MaxPooling2D, Dropout, Activation, Concatenate
from tensorflow.keras.layers import Add, Multiply, Input, Conv2DTranspose,LeakyReLU, ReLU



class ConvLayer():
    def __init_(self,
                kernel_size
                padding
                initializer):

        self.kernel_size=kernel_size
        self.padding=padding
        self.initializer=initializer


    def __call__(f):
        return Conv2D(
                filters=f, 
                kernel_size=self.kernel_size,
                padding=self.padding, 
                kernel_initializer=self.initializer
                )


class UpLayer():
    def __init_(self,
                kernel_size
                padding
                initializer
                activation,
                layer_type):
        
        self.kernel_size=kernel_size
        self.padding=padding
        self.initializer=initializer
        self.activation=activation
        self.layer_type=layer_type


    def __call__(self, f):
        if self.layer_type=='upsampling':
            layer=UpSampling2D((2,2))
        elif self.layer_type=='transpose':
            layer=Conv2DTranspose(
                            f,
                            kernel_size=self.kernel_size,
                            activation=self.activation, 
                            strides=(2,2), 
                            padding=self.padding
                            )
        return layer



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

