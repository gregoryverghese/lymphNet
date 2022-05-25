import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Dropout, Input, concatenate, LeakyReLU, Concatenate, Activation


class MobileUnet():
    def __init__(self, filters=[16, 32, 48, 64], 
                 final_activation='sigmoid', 
                 n_output=1,
                 up_type='upsampling'):

        self.filters = [16, 32, 48, 64]
        self.final_activation=final_activation
        self.n_output=n_output
        self.up_type=up_type


    def build(self):

        inputs = Input(shape=(None, None, 3), name="input_image")
        encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=0.35)
        skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
        encoder_output = encoder.get_layer("block_13_expand_relu").output
        x = encoder_output

        for i in range(1, len(skip_connection_names)+1, 1):
            x_skip = encoder.get_layer(skip_connection_names[-i]).output
            x = UpSampling2D((2, 2))(x)
            x = Concatenate()([x, x_skip])

            x = Conv2D(self.filters[-i], (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(self.filters[-i], (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

        x = Conv2D(1, (1, 1), padding="same")(x)
        x = Activation(self.finalActivation)(x)

        model = Model(inputs, x)
        
        return model
