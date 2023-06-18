import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.layers import AveragePooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
#from tensorflow.keras.initializers import HeNormal


class DeepLabV3Plus():
    def __init__(self,
                 filters=[32,64,128, 256,512], 
                 final_activation='sigmoid',
                 dropout=0, 
                 n_output=1,
                 dims=1024):
        self.numClasses=n_output
        self.dims=dims
    
    def convBlock(self,bInput,nFilters=256,kSize=3,dilation=1,padding="same", bias=False):
        #initializer=tf.keras.initializers.HeUniform(seed=None)
        initializer=tf.keras.initializers.he_uniform()
        x = Conv2D(nFilters,kernel_size=kSize,dilation_rate=dilation,padding="same",
                          use_bias=bias,kernel_initializer=initializer)(bInput)
        x = BatchNormalization()(x)
        return tf.nn.relu(x)


    def DilatedSpatialPyramidPooling(self,dsppInput):
        dims = dsppInput.shape
        #dims = (256,256)
        print('greg1',dims)
        x = AveragePooling2D(pool_size=(dims[-3],dims[-2]))(dsppInput)
        x = self.convBlock(x, kSize=1, bias=True)
        print('greg2',x.shape)
        outPool = UpSampling2D(size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear")(x)
        out1 = self.convBlock(dsppInput, kSize=1,dilation=1)
        out6 = self.convBlock(dsppInput,kSize=3,dilation=6)
        out12 = self.convBlock(dsppInput,kSize=3,dilation=12)
        out18 = self.convBlock(dsppInput,kSize=3,dilation=18)
        x = Concatenate(axis=-1)([outPool,out1,out6,out12,out18])
        output = self.convBlock(x,kSize=1)
        return output


    def build(self):
        modelInput = Input(shape=(1024, 1024, 3))
        resnet50 = ResNet50(weights="imagenet",include_top=False,input_tensor=modelInput)
        x = resnet50.get_layer("conv4_block6_2_relu").output
        x = self.DilatedSpatialPyramidPooling(x)
        inputA = UpSampling2D(size=(self.dims // 4 // x.shape[1], self.dims // 4 // x.shape[2]),interpolation="bilinear")(x)
        inputB = resnet50.get_layer("conv2_block3_2_relu").output
        inputB = self.convBlock(inputB, nFilters=48, kSize=1)
        x = Concatenate(axis=-1)([inputA, inputB])
        x = self.convBlock(x)
        x = self.convBlock(x)
        x = UpSampling2D(size=(self.dims // x.shape[1], self.dims // x.shape[2]),interpolation="bilinear")(x)
        modelOutput = Conv2D(self.numClasses,kernel_size=(1, 1),padding="same")(x)
        return Model(inputs=modelInput,outputs=modelOutput)

