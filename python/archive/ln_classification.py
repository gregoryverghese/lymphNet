import os
import tensorflow as tf
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
import numpy as np
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
AUGMENT = {'rescale':1./255}
LOSS = 'binary_crossentropy'
EPOCHS = 15
MODELNAMES = ['VGG16', 'Inception', 'ResNet50']
BASEDIR = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/classification'
#BASEDIR = 'test3'
RATIO = 0.8
WEIGHTS = 'imagenet'
METRICS = [getF1, getRecall, getPrecision]


def getTransferModel(methods, imgSize, weights, channels=3):

    transfer = TransferLearning(imgSize, weights, channels)
    models = [getattr(transfer,'get'+f + 'Model')() for f in methods]
    lastNames = [transfer.getLayerNames(m) for m in models]
    models = [transfer.freezeLayers(m) for m in models]


    print(lastNames)

    lastLayers = [m.get_layer(n) for m, n in zip(models,lastNames)]
    lastOutputs = [l.output for l in lastLayers]
    finalLayers = [transfer.getFinalLayer(l, 'sigmoid',  1, 128, drop=0.2) for l in lastOutputs]
    models = [ Model(m.input, x) for m, x in zip(models, finalLayers)]

    return models

'''
save model down
'''
def saveModel(model, modelName, filePath='~/models/'):

    #model.save(filePath+modelName+'_bin_1.h5')

    model_json = model.to_json()

    with open(modelName+'.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(modelName + 'Weights.h5')


'''
save history down
'''
def saveHistory(history, name):

    with open(name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

'''
load save history. Note returns as dictionary
'''
def loadHistory(name):

    with open('~/models/'+name, 'rb') as f:
        history = pickle.load(f)

    return history


'''
compile keras model
'''
def compileModel(model, loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['acc']):
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
def getGenerators(trainDir, validationDir, augmented, targetSize, batchSize, classMode, shuffle=True):

    trainDataGen = ImageDataGenerator(**augmented)
    validationDataGen = ImageDataGenerator(rescale=1./255)

    trainGenerator = trainDataGen.flow_from_directory(
    trainDir,
    target_size=targetSize,
    batch_size=batchSize,
    class_mode=classMode
    )

    validationGenerator = validationDataGen.flow_from_directory(
    validationDir,
    target_size=targetSize,
    batch_size=batchSize,
    shuffle=shuffle,
    class_mode='binary'
    )

    return trainGenerator, validationGenerator


def testGenerator(testSinus, testGerminal, trainGenerator):

    #testDirectories = glob.glob(baseDir+'/*')

    sinusFiles = glob.glob(os.path.join(testSinus, '*'))
    germinalFiles = glob.glob(os.path.join(testGerminal, '*'))
    #follicleFiles = glob.glob(os.path.join(validFollicle, '*'))

    germinalImgs = utilities.getImages(testGerminal, trainGenerator.class_indices['GERMINAL'], (224, 224))
    sinusImgs = utilities.getImages(testSinus, trainGenerator.class_indices['SINUS'], (224, 224))
    #follicleImgs = getImages(validFollicle, trainGenerator.class_indices['FOLLICLE'], (224, 224))

    germinalImages = [next(germinalImgs) for i in range(len(germinalFiles))]
    sinusImages = [next(sinusImgs) for i in range(len(sinusFiles))]
    #follicleImages = [next(follicleImgs) for i in range(len(follicleFiles))]
    images = sinusImages + germinalImages
    random.shuffle(images)

    imgArrs, labels = zip(*images)

    return imgArrs, labels


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

    if len(os.listdir(trainSinus)) == 0:
        numSplitSinus = utilities.MoveFiles(BASEDIR, 'SINUS', trainSinus, validSinus, testSinus, RATIO)
        numSplitGerminal = utilities.MoveFiles(BASEDIR, 'GERMINAL', trainGerminal, validGerminal, testGerminal  , RATIO)
        print('Sinus \n train: {} test {} valid {} \n Germinal \n train: {} test {} valid {}'.format(numSplitSinus[0],
                        numSplitSinus[1], numSplitSinus[2], numSplitGerminal[0], numSplitGerminal[1], numSplitGerminal[2]))


    trainGenerator, validGenerator = getGenerators(trainDir, validationDir, AUGMENT, (IMGSIZE, IMGSIZE), BATCH, 'binary')
    models = getTransferModel(MODELNAMES, IMGSIZE, WEIGHTS)
    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    compile = lambda x: compileModel(x, loss=LOSS, optimizer=adam, metrics=METRICS)
    models = list(map(compile, models))

    fit = lambda x: x.fit_generator(trainGenerator,
                                        steps_per_epoch=trainGenerator.samples//trainGenerator.batch_size,
                                        epochs=15,
                                        validation_data=validGenerator,
                                        validation_steps=validGenerator.samples//validGenerator.batch_size,
                                        verbose=1)

    histories = list(map(fit, models))

    for i in range(len(models)):
        print('i am here')
        saveModel(models[i], MODELNAMES[i])
        saveHistory(histories[i], MODELNAMES[i])

    imgArrs, labels = testGenerator(testSinus, testGerminal, trainGenerator)
    images = np.vstack(imgArrs)
    classes = [m.predict(images, batch_size=BATCH) for m in models]
    classes = [np.argmax(c, axis=1) for c in classes]
    print(classes[0])
    resultsDict = {'VGG16': classes[0], 'INCEPTION': classes[1], 'RESNET': classes[2], 'TRUE': labels}
    results = pd.DataFrame(resultsDict)
    results.to_csv('results.csv')
    results.to_csv('ln_classification_bin_1.csv')

if __name__ == '__main__':
    getClassification()
