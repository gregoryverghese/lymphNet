#!/usr/bin/env python3

'''
ln_segmentation.py: main module that sets up analysis, models
parameters and calls training and prediction scripts
'''

import os
import sys
import csv
import glob
import pickle
import random
import argparse
import json
import datetime

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from prettytable import PrettyTable
from tensorflow import keras
from skimage.transform import resize
from skimage import img_as_bool
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import LearningRateScheduler
#from tensorflow.keras.utils import multi_gpu_model

import distributed_train
from models import fcn8
from models import unet
from models import mobile
from models import resunet
from models import resunet_a
from models import unet_mini
from models import atten_unet
from models import multiscale
from models import multi_atten
from models import deeplabv3
from data.tfrecord_read import TFRecordLoader
from utilities import decay_schedules
from utilities.evaluation import diceCoef
from predict import WSIPredictions, PatchPredictions
from utilities.utils import getTrainCurves
from preprocessing.calculate_classweights import calculateWeights
from utilities.custom_loss_classes import WeightedBinaryCrossEntropy, DiceLoss, WeightedCategoricalCrossEntropy

FUNCMODELS={
            'unet':unet.UnetFunc,
            'unetmini':unet_mini.UnetMiniFunc,
            'attention':atten_unet.AttenUnetFunc,
            'multiscale':multiscale.MultiScaleUnetFunc,
            'multiatten':multi_atten.MultiAttenFunc,
            'resunet':resunet.ResUnetFunc,
            'fcn8':unet.UnetFunc,
            'mobile':mobile.MobileUnetFunc,
            'deeplabv3plus':deeplabv3.DeepLabV3Plus
            }


SUBCLASSMODELS={
              'unet':unet.UnetSC,
              'unetmini':unet_mini.UnetMiniSC,
              'attention':atten_unet.AttenUnetSC,
              'multiscale':multiscale.MultiScaleUnetSC,
              'resunet':resunet.ResUnetFunc,
              'fcn8':unet.UnetFunc
               }            


def dataLoader(path,config):
    trainPath = os.path.join(path,'train','*.tfrecords')
    trainFiles = glob.glob(trainPath)
    trainLoader=TFRecordLoader(trainFiles,config['imageDims'],'train',config['tasktype'])    
    validPath = os.path.join(path,'validation','*.tfrecords')
    validFiles = glob.glob(validPath)
    validLoader=TFRecordLoader(validFiles,config['imageDims'],'valid',config['tasktype'])
    return trainLoader,validLoader


def main(args):
    '''
    sets up analysis based on config file. Following design choices:

    1. model
    2. optimizer and optimizer decay
    3. loss function

    load tfrecords data, calls custom training and prediction scripts. 
    Parameters and model are saved down
    
    :param args: command line arguments
    :returns result: avg dice and iou score
    '''    
    with open(args['configfile']) as jsonFile:
        config = json.load(jsonFile)

    currentDate = str(datetime.date.today())
    currentTime = datetime.datetime.now().strftime('%H:%M')
    path = os.path.join(args['recordpath'],args['recordDir'])
    trainLoader,validLoader=dataLoader(path,config)

    print('\n'*4+'-'*25 + 'Lymphnode segmentation' +'-'*25+'\n'*2)
    test='unet'
    print(f'network:{test} \
          \nfeature:{config["feature"]} \
          \nmag:{config["magnification"]} \
          \nloss:{config["loss"]} \
          \nweights:{config["weights"]} \
          \naugmentation:{config["augmentation"]}')

    #num gpus available
    devices = tf.config.experimental.list_physical_devices('GPU')
    devices = [x.name.replace('/physical_device:', '') for x in devices] 
    #devices = ['/device:GPU:{}'.format(i) for i in range(multiDict['num'])]

    nnParams={ #'filters':filters,
               #'finalActivation':finalActivation,
               'nOutput':nClasses,
               'dims':imgDims
               #'upTypeName':upTypeName
                }

    #trainDataset=trainDataLoader.getShards(config['batchSize'])

    strategy = tf.distribute.MirroredStrategy(devices)
    with strategy.scope():
        #get optimizer and decay schedule if applicable
        if decaySchedule['keras'] is not None:
            boundaries = [b*trainSteps for b in decaySchedule['decayparams']['boundaries']]
            values = decaySchedule['decayparams']['values']
            lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries,values)
        elif decaySchedule['custom']['method'] is not None:
            scheduleKwargs = decaySchedule['custom']['decayparams']
            scheduler = getattr(decay_schedules, decaySchedule['custom']['method'])(**scheduleKwargs)
            callbacks += [LearningRateScheduler(scheduler)]
            scheduler.plot(epoch, modelName + ': ' + decaySchedule + ' learning rate')

        if optimizername=='adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        elif optimizername=='Nadam':
            optimizer = keras.optimizers.NAdam(**optKwargs)
        elif optimizername=='SGD':
            optimizer=keras.optimizers.SGD(**optKwargs)
        else:
            raise ValueError('No optimizer selected, please update config file')

        if loss=='weightedBinaryCrossEntropy':
            lossObject = WeightedBinaryCrossEntropy(weights[0])
        elif loss=='weightedCategoricalCrossEntropy':
            lossObject = WeightedCategoricalCrossEntropy(weights)
        elif loss=='diceloss':
            lossObject = DiceLoss()
        else:
            raise ValueError('No loss requested, please update config file')

        with tf.device('/cpu:0'):
            if api=='functional':
                model=FUNCMODELS[modelname](**nnParams)
            elif api=='subclass':
                model=SUBCLASSMODELS[modelname](**nnParams)

            model=model.build()
            #model = sm.Unet('vgg19', input_shape=(None, None, 3))
            if model is None:
                raise ValueError('No model requested, please update config file')
        print(model)
        table = PrettyTable(['Model', 'Loss', 'Optimizer', 'Devices','DecaySchedule'])
        table.add_row([modelname, loss, optimizername, len(devices),decaySchedule['keras']])
        print(table)
        
        if trainingapi=='custom':
            #call distributed training script to allow for training on multiple gpus
            trainDistDataset = strategy.experimental_distribute_dataset(trainDataset)
            validDistDataset = strategy.experimental_distribute_dataset(validDataset)
                                                  
            train = distributed_train.DistributeTrain(epoch, model, optimizer, 
                                                      lossObject, batchSize, strategy, 
                                                      trainSteps, validSteps, imgDims, 
                                                      stopthresholds, modelName, currentDate, 
                                                      currentTime, tasktype)

            model, history = train.forward(trainDistDataset, validDistDataset)
            
    if trainingapi=='keras':

        #model = multi_gpu_model(model, gpus=multiDict['num'])
        model.compile(optimizer=optimizer,loss=lossObject,metrics=[diceCoef])
        model.fit(trainDataset, epochs=epoch, steps_per_epoch=trainSteps, validation_data=validDataset)


    #save models down to model folder along with training history
    if api == 'subclass':
        model.save(os.path.join(outModelPath, modelName + '_' + currentTime), save_format='tf')

    elif api == 'functional': 
        model.save(os.path.join(outModelPath, modelName + '_' + currentTime + '.h5'))

    with open(os.path.join(outModelPath,  modelName+'_'+ currentTime + '_history'), 'wb') as f:
        pickle.dump(history, f)

    #call predict on individual patches and entire annotated wsi region
    #patchpredict = PatchPredictions(model, modelName, batchSize, currentTime, currentDate, activationthreshold)
    #patchpredict(testdataset, os.path.join(outPath, 'predictions'))

    #outpath='/home/verghese/breastcancer_ln_deeplearning/output/predictions/wsi'
    wsipredict = WSIPredictions(model, modelName, feature,
                                magnification,step, step,
                                activationthreshold, currentTime,
                                currentDate, tasktype, normalize,
             valuation.py                    normalizeParams)

    result = wsipredict(os.path.join(recordsPath, 'tfrecords_wsi'), outPath)
    #ToDo: replace generic trainmetric with metric name from config file
    getTrainCurves(history,'trainloss','valloss',outCurvePath,modelName+'_'+currentTime)
    getTrainCurves(history,'trainmetric', 'valmetric', outCurvePath,modelName+'_'+currentTime)

    #finally save the config for this file with the model and predictions
    configPath=os.path.join(outModelPath,modelName+'_'+currentTime+'_config.json')
    with open(configPath, 'w') as jsonFile:
        json.dump(params, jsonFile)

    return result


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-rp', '--recordpath', required=True, help='path to tfrecords')
    ap.add_argument('-rd', '--recordDir', required=True, help='directory for the tfrecords dataset')    
    ap.add_argument('-op', '--outpath', required=True, help='output path for predictions')
    ap.add_argument('-cp', '--checkpointpath', required=True, help='path for checkpoint files')
    ap.add_argument('-cf', '--configfile', help='file containing parameters')
    ap.add_argument('-mn', '--modelname', help='name of neural network model')

    args = vars(ap.parse_args())
    
    outModelPath = os.path.join(args['outpath'],'models')
    os.makedirs(outModelPath,exist_ok=True)
    outModelPath = os.path.join(args['outpath'],'models',currentDate)
    os.makedirs(outModelPath,exist_ok=True)
    outCurvePath = os.path.join(args['outpath'],'curves')
    os.makedirs(outCurvePath,exist_ok=True)
    outCurvePath = os.path.join(args['outpath'],'curves',currentDate)
    os.makedirs(outCurvePath,exist_ok=True)
    outPredictPath=os.path.join(args['outpath'],'predictions')
    os.makedirs(outPredictPath,exist_ok=True)

    main(args)
