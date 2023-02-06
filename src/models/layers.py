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


    #Keras requires either dilation or strides to be 1 so we cannot replicate Nikhil
    def __call__(self, f, dilation=1):
        return Conv2D(
                filters=f, 
                kernel_size=self.kernel_size,
                padding=self.padding, #may need to incl padding as an argument 
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

# =DoubleConv
# HR - 5/9/22 - added scale factor to call this from multiblock
def conv_block(x,
               f,
               conv_layer,
               normalize=True,
               scale=1, 
               drop=False):

    x = conv_layer(f, scale)(x)
    #we have to pool because Keras cannot handle both dilation AND stride >1 at same time 
    x = MaxPooling2D(pool_size=(scale,scale))(x) if scale > 1 else x
    x = BatchNormalization()(x) if normalize else x
    x = ReLU()(x)

    #only need to scale on the first conv layer 
    x = conv_layer(f)(x)
    x = BatchNormalization()(x) if normalize else x
    x = ReLU()(x)
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

# = Down_MR block
### HR - 5/9/22 - removed MaxPool to mirror the conv_block structure - must be called outside the block
### now the additional MaxPool (as a stride replacement) occurs inside the conv_block
def multi_block(x,
                f,
                conv_layer,
                up_layer,
                normalize=True
               ):
    out_f=f/3

    x1 = conv_block(x, out_f, conv_layer, normalize, scale=1)

    x2 = conv_block(x, out_f, conv_layer, normalize, scale=2)
    x2 = up_layer(pool=(2,2))(x2) #return x2 to same dimensions as x1 

    x3 = conv_block(x, out_f, conv_layer, normalize, scale=4)
    x3 = up_layer(pool=(4,4))(x3)

    print("\n*** HOLLY - just before concat")
    x = Concatenate()([x1,x2,x3])

    print("\n*** HOLLY - AFTER concat")
    return x






