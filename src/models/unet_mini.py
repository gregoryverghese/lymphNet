import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model


class UnetMini():
    def __init__(self, 
                 filters=[32,64,128,256], 
                 final_Activation='sigmoid', 
                 activation='relu',
                 n_output=1, 
                 kernel_size=(3,3), 
                 pool=(2,2), 
                 initializer='glorot_uniform',
                 dropout=0, 
                 normalize=True, 
                 padding='same', 
                 dtype='float32'):

        self.filters = filters
        self.activation = activation
        self.final_activation = final_activation
        self.n_output = n_output
        self.kernel_size = kernel_size
        self.pool = pool
        self.dropout=dropout
        self.normalize = normalize
        self.padding = padding
        self.initializer = initializer
        self.dtype = dtype


    @property
    def conv_layer(self):
        return ConvLayer(
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


    def bridge(self, x, f, kSize=(3, 3)):

        x = Conv2D(f, kSize, padding=self.padding)(x)
        x = BatchNormalization()(x) if self.normalize else x
        x = Conv2D(f, kSize, padding=self.padding)(x)
        x = BatchNormalization()(x) if self.normalize else x

        return x


    def encoder(self, x):

        e1 = conv_block(x, self.filters[0], self.conv_layer)
        p1 = MaxPooling2D((2,2))(e1)
        e2 = conv_block(p1, self.filters[1], self.conv_layer)
        p2 = MaxPooling2D((2,2))(e2)
        e3 = conv_block(p2, self.filters[2], self.conv_layer)
        p3 = MaxPooling2D((2,2))(e3)
        bridge = conv_block(p4, self.filters[3], self.conv_layer)
 
        return e1,e2,e3,bridge


    def decoder(self,e1,e2,e3,bridge):

        d3 = self.up_layer(self.filters[3])(bridge)
        d3 = Concatenate()([e3, d3])
        d3 = conv_block(d3, self.filters[2], self.conv_layer)
        
        d2 = self.up_layer(self.filters[2])(d4)
        d2 = Concatenate()([e2, d3])
        d2 = conv_block(d3, self.filters[1], self.conv_layer)

        d1 = self.up_layer(self.filters[1])(d3)
        d1 = Concatenate()([e1, d2])
        d1 = conv_block(d2, self.filters[0], self.conv_layer)

        return d1


    def build(self):

        tensor_input = Input((None, None, 3))
        e1,e2,e3,bridge = self.encoder(tensor_input)
        d1 = self.decoder(e1,e2,e3,bridge)
        final_map = Conv2D(self.n_output, (1, 1), activation=self.final_activation)(d1)
        model = Model(tensor_input, final_map)

        return model
