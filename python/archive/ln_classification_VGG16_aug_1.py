import os
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
import shutil
import random
import glob
import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image
import pickle
import json
import random
import pandas
import utilities
from classification_transfer import TransferLearning
from evaluation import getF1, getPrecision, getRecall


BATCH = 2
IMGSIZE = 224
AUGMENT = {
            'rescale':1./255,
            'rotation_range':360,
            'zoom_range':[0.5,1.5],
            'horizontal_flip':True,
            'vertical_flip':True
            }
LOSS = 'binary_crossentropy'
EPOCHS = 15
MODELNAMES = ['VGG16']
BASEDIR = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/classification'
RATIO = 0.8
WEIGHTS = 'imagenet'
METRICS = ['acc']


def getTransferModel(methods, imgSize, weights, channels=3):

    transfer = TransferLearning(imgSize, weights, channels)
    models = [getattr(transfer,'get'+f + 'Model')() for f in methods]
    lastNames = [transfer.getLayerNames(m) for m in models]
    models = [transfer.freezeLayers(m) for m in models]
    lastLayers = [m.get_layer(n) for m, n in zip(models,lastNames)]
    lastOutputs = [l.output for l in lastLayers]
    finalLayers = [transfer.getFinalLayer(l, 'sigmoid',  1, 128, drop=0.2) for l in lastOutputs]
    models = [ Model(m.input, x) for m, x in zip(models, finalLayers)]

    return models

'''
save model down
'''
def saveModel(model, modelName, filePath='/home/verghese/models/'):

    #model.save(filePath+modelName+'_bin_1.h5')

    model_json = model.to_json()

    with open(filePath+modelName+'.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(filePath+modelName + 'Weights.h5')


'''
save history down
'''
def saveHistory(history, name):

    with open('/home/verghese/models/'+name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

'''
load save history. Note returns as dictionary
'''
def loadHistory(name):

    with open('/home/verghese/models/'+name, 'rb') as f:
        history = pickle.load(f)

    return history


'''
compile keras model
'''
def compileModel(model, loss='binary_crossentropy', optimizer='adam', metrics=['acc']):
    model.compile(loss=loss, optimizer=optimizer,  metrics=metrics)
    return model


'''
fit model to training data using train generator
'''
def fitModel(model, trainGenerator, validationGenerator, callbacks, epochs=10):

    history = model.fit_generator(
    trainGenerator,
    steps_per_epoch=trainGenerator.samples//trainGenerator.batch_size,
    epochs=epochs,
    validation_data=validationGenerator,
    validation_steps=validationGenerator.samples//validationGenerator.batch_size,
    verbose=1)

    return history, model


'''
returns predictions for each train instance
KNOWN BUG IN PREDICT GENERATOR relating to batches
'''
def predictGenerator(model, validationGenerator):

    validationGenerator.reset()
    yPred = model.predict_generator(validationGenerator, validationGenerator.samples/validationGenerator.batch_size, workers=0)
    yPred = np.argmax(yPred, axis=1)

    return yPred

'''
predict class for each test instance
'''
def predictClasses(model, images, batchSize=32):
    return model.predict_classes(images, batch_size=batchSize)



'''
return training and validation generators. Generator control flow of data to
model. Automatically  generates labels based on folder structure
'''
def getGenerators(trainDir, validationDir,testDir, augmented, targetSize, batchSize, classMode, shuffle=True):

    trainDataGen = ImageDataGenerator(**augmented)
    validationDataGen = ImageDataGenerator(rescale=1./255)
    testDataGen = ImageDataGenerator(rescale=1./255)

    trainGenerator = trainDataGen.flow_from_directory(
    trainDir,
    target_size=targetSize,
    batch_size=batchSize,
    class_mode=classMode,
    )

    validationGenerator = validationDataGen.flow_from_directory(
    validationDir,
    target_size=targetSize,
    batch_size=batchSize,
    shuffle=shuffle,
    class_mode=classMode
    )

    testGenerator = testDataGen.flow_from_directory(
    testDir,
    target_size = targetSize,
    batch_size = 1,
    class_mode = classMode,
    shuffle = False)

    return trainGenerator, validationGenerator, testGenerator


def getClassWeights(trainGenerator):

    classWeights = class_weight.compute_class_weight('balanced', np.unique(trainGenerator.classes), trainGenerator.classes)
    return {k:v for k, v in zip(trainGenerator.class_indices.values(), classWeights)}


def getClassification():

    trainDir = os.path.join(BASEDIR, 'train')
    validationDir = os.path.join(BASEDIR, 'validation')
    testDir = os.path.join(BASEDIR, 'test')
    trainSinus = os.path.join(trainDir, 'SINUS')
    trainGerminal = os.path.join(trainDir, 'GERMINAL')
    testSinus = os.path.join(testDir, 'SINUS')
    testGerminal = os.path.join(testDir, 'GERMINAL')
    validSinus = os.path.join(validationDir, 'SINUS')
    validGerminal = os.path.join(validationDir, 'GERMINAL')

    if len(os.listdir(trainSinus)) == 1:
        numSplitSinus = utilities.MoveFiles(BASEDIR, 'SINUS', trainSinus, validSinus, testSinus, RATIO)
        numSplitGerminal = utilities.MoveFiles(BASEDIR, 'GERMINAL', trainGerminal, validGerminal, testGerminal  , RATIO)
        print('Sinus \n train: {} test {} valid {} \n Germinal \n train: {} test {} valid {}'.format(numSplitSinus[0],
                        numSplitSinus[1], numSplitSinus[2], numSplitGerminal[0], numSplitGerminal[1], numSplitGerminal[2]))

    trainGenerator, validGenerator, testGenerator = getGenerators(trainDir, validationDir,testDir,  AUGMENT, (IMGSIZE, IMGSIZE), BATCH, 'binary')
    models = getTransferModel(MODELNAMES, IMGSIZE, WEIGHTS)
    compile = lambda x: compileModel(x, loss=LOSS, optimizer='adam', metrics=METRICS)
    models = list(map(compile, models))

    classWeights = getClassWeights(trainGenerator)
    
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    fit = lambda x: x.fit_generator(trainGenerator,
                                        steps_per_epoch=trainGenerator.samples//trainGenerator.batch_size,
                                        epochs=1,
                                        validation_data=validGenerator,
                                        validation_steps=validGenerator.samples//validGenerator.batch_size,
                                        class_weight= classWeights,
                                        verbose=1)

    histories = list(map(fit, models))

    for i in range(len(models)):
        saveModel(models[i], MODELNAMES[i]+'_aug')
        saveHistory(histories[i], MODELNAMES[i]+'_aug')

    predictions = list(map(lambda x: x.predict_generator(testGenerator, steps=testGenerator.samples),models))
    classPredictions  = list(map(lambda x: (np.array(x)  > 0.5).astype(np.int), predictions))

    resultsDict = {'VGG16': classPredictions[0], 'TRUE': testGenerator.classes}
    results = pd.DataFrame(resultsDict)
    results.to_csv('/home/verghese/results_VGG16_aug.csv')
    results.to_csv('/home/verghese/output/ln_classification_bin_1_VGG1_˙6_aug.csv')

if __name__ == '__main__':
    getClassification()
