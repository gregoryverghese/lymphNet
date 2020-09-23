import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50


class TransferLearning():
    def __init__(self, imgSize, weights, channels=3):
        self.imgSize = imgSize
        self.weights = weights
        self.channels = channels

    '''
    Initialize inception model
    '''
    def getInceptionModel(self):
        return InceptionV3(input_shape=(self.imgSize, self.imgSize, self.channels), include_top=False, weights=self.weights)


    '''
    Initialize VGG16 model
    '''
    def getVGG16Model(self):
        return VGG16(input_shape=(self.imgSize, self.imgSize, self.channels), include_top=False, weights=self.weights)


    '''
    Initialize Resnet model
    '''
    def getResNet50Model(self):
        return ResNet50(input_shape=(self.imgSize, self.imgSize, self.channels), include_top=False, weights=self.weights)


    '''
    Freeze all layers in the model
    '''
    def freezeLayers(self, model):

        for layer in model.layers:
            layer.trainable = False

        return model


    '''
    return last layer and the output of the last layer
    '''
    def getLastLayer(self, model, name):

        lastLayer = model.get_layer(name)
        lastOutput = lastLayer.output

        return lastOutput, lastLayer


    '''
    get names of each layer in the model
    '''
    def getLayerNames(self, model):
        return [layer.name for layer in model.layers][-1]


    '''
    Build final layers for training in model
    '''
    def getFinalLayer(self, lastOutput, activation, finalLayer, denseLayer, drop=0.2):

        print(lastOutput)

        x = tf.keras.layers.Flatten()(lastOutput)
        x = tf.keras.layers.Dense(denseLayer, activation='relu')(x)
        x = tf.keras.layers.Dropout(drop)(x)
        x = tf.keras.layers.Dense(finalLayer, activation=activation)(x)

        return x
