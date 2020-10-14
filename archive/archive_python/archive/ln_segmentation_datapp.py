import os
import csv
import cv2
import glob
import numpy as np
import pickle
import random
import argparse
import json
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow import keras
from skimage.transform import resize
from skimage import img_as_bool
from tensorflow.keras import backend as K

from resunet_multi import Unet
from fcn8 import FCN
from utilities import saveModel, saveHistory
from evaluation import dice_coef_loss, dice_coef
from custom_datagenerator_three import DataGenerator
from custom_loss_functions import weightedCatXEntropy
from image_mask_loaded_2 import ImageMaskProcessor


def calculateWeights(trainGenerator, trainSteps):

    for i in range(trainSteps):
        _, m = trainGenerator.__getitem__(i)
        mask = np.argmax(m, axis=3)
        labels.append(mask.reshape(-1))

    labels = [l.tolist() for l in labels]
    labels = itertools.chain(*labels)
    weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)

    return weights


def getPrediction(model, validGenerator, validIds):

    steps = len(validIds)//validGenerator.batchSize

    for i in range(0, steps):
        x, y = validGenerator.__getitem__(i)
        y[y==1]=255
        masks.append(y)
        yPred = model.predict(x)
        yPred = np.argmax(yPred, axis=3)

        for img in yPred:
                x, y = validGenerator.__getitem__(i)
                y[y==1]=255
                masks.append(y)
                yPred = model.predict(x)
                yPred = np.argmax(yPred, axis=3)


def trainSegmentationModel(args):

    basePath = args['basepath']
    imageDir = args['imagedir']
    maskDir = args['maskdir']

    if args['weightfile'] is not None:
        with open(args['weightfile'], 'r') as txtFile:
            weights = list(csv.reader(txtFile, delimiter=','))
    weights= [int(float(w)) for w in weights[0]]
    with open(args['paramfile']) as jsonFile:
        params = json.load(jsonFile)

    print(params['nClasses'])

   # if args['model'] == 'unet':
    unet =  Unet(int(params['imageDims']), nOutput = int(params['nClasses']), finalActivation=params['final'])
    model = unet.ResUnet()
   # elif args['model'] == 'fcn8':
        #fcn = FCN(int(params['imageDims']), nClasses = int(params['nClasses']), finalActivation=params['final'])
        #model = fcn.getFCN8()

    epoch = int(params['epoch'])
    ratio = float(params['ratio'])

    imagePath = os.path.join(basePath, imageDir)
    maskPath = os.path.join(basePath, maskDir)

    numFiles = len(glob.glob(os.path.join(imagePath, '*')))
    
    #if args['weightfile'] is None:
        #calculateWeights
    
    datagen = ImageMaskProcessor(3000, imagePath, maskPath, 224, weights)
    images, masks = datagen.load()
    
    adam = keras.optimizers.Adam()
    model.compile(optimizer=adam, loss=weightedCatXEntropy, metrics=[dice_coef])

    history = model.fit(images, masks,verbose=1, epochs=epoch, validation_split=0.1)

    model.save(args['name']+'11.h5')

    saveModel(model, args['name'])
    saveHistory(history, args['name']+'_hist')

    #getPrediction(model, validGenerator, validIds)

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-bp', '--basepath', required=True, help='path to image and mask directories')
    ap.add_argument('-ip', '--imagedir', required=True, help='path to image directory')
    ap.add_argument('-mp', '--maskdir', required=True, help='path to image directory')
    ap.add_argument('-m', '--model', required=True, help='neural network model to use')
    ap.add_argument('-n', '--name', required=True, help='name to save the model with')
    ap.add_argument('-wf', '--weightfile', help='file containing list of class weights for unbalanced datasets')
    ap.add_argument('-pf', '--paramfile', help='file containing parameters')

    args = vars(ap.parse_args())

    trainSegmentationModel(args)
