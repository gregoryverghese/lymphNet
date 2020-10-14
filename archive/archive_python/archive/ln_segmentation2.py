import os
import sys
import csv
import cv2
import glob
import numpy as np
import pickle
import random
import argparse
import json
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tensorflow import keras
from skimage.transform import resize
from skimage import img_as_bool
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import multi_gpu_model

import fcn8
import unet2
import unetsc
import resunet
import unet_mini
import atten_unet
import distributed_train
import tfrecord_read
import decay_schedules
from evaluation import diceCoef
from predict import WSIPredictions, PatchPredictions
from utilities import getTrainMetric
from calculate_classweights import calculateWeights
from custom_loss_classes import WeightedBinaryCrossEntropy, DiceLoss


def trainSegmentationModel(args):

    recordsPath = args['recordpath']
    outPath = args['outpath']
    checkpointPath = args['checkpointpath']
    configFiles = args['configfile']

    with open(configFiles) as jsonFile:
        params = json.load(jsonFile)

    feature = params['feature']
    model = params['model']
    optimizer = params['optimizer']['method']
    loss =  params['loss']
    dropout = params['dropout']
    batchSize = params['batchSize']
    epoch = params['epoch']
    ratio = params['ratio']
    augment = params['augmentation']
    weights = params['weights']
    loss =  params['loss']
    metric = params['metric']
    modelName = params['modelname']
    optKwargs = params['optimizer']['parameters']
    decaySchedule = params['optimizer']['decaymethod']
    normalize = params['normalize']
    padding = params['padding']
    multiDict = params['multi']
    multi = multiDict['flag']
    imgDims = params['imageDims']
    nClasses = params['nClasses']
    final = params['final']
    filters = params['filters']
    api = params['modelapi']
    magnification = params['magnification']
    traintype = params['trainapi']

    trainFiles = glob.glob(os.path.join(recordsPath,'train','*.tfrecords'))
    validFiles = glob.glob(os.path.join(recordsPath,'validation','*.tfrecords'))
    testFiles = glob.glob(os.path.join(recordsPath,'test','*.tfrecords'))

    trainNum = tfrecord_read.getRecordNumber(trainFiles)
    validNum = tfrecord_read.getRecordNumber(validFiles)
    testNum = tfrecord_read.getRecordNumber(testFiles)

    currentDate = datetime.date.today()
    currentTime = datetime.datetime.now()

    trainSteps = np.ceil(trainNum/batchSize) if trainNum<batchSize else np.floor(trainNum/batchSize)
    validSteps = np.ceil(validNum/batchSize) if validNum<batchSize else np.floor(validNum/batchSize)
    testSteps = np.ceil(testNum/batchSize) if testNum<batchSize else np.floor(testNum/batchSize)

    if weights is None:
        weights = calculateWeights('test', 'test', 'test', nClasses)
 
     
    print('{} Image segmentation of uinvolved lymph nodes with a {} model {}'.format(('-'*50, model, '-'*50))
    print('TrainNum: {}'.format(trainNum))
    print('ValidNum: {}'.format(validNum))
    print('TestNum: {}'.format(testNum))
    print('TrainSteps: {}'.format(trainSteps))
    print('TestSteps: {}'.format(testSteps))
    print('ValidSteps: {}'.format(validSteps))
    print('ValidNum: {}'.format(validNum))
    print('TestNum: {}'.format(testNum))
    print('Weights: {}'.format(weights))

    trainDataset = tfrecord_read.getShards(trainFiles, imgDims=imgDims, batchSize=batchSize, dataSize=trainNum,  augmentations=augment)
    validDataset = tfrecord_read.getShards(validFiles, imgDims=imgDims, batchSize=batchSize, dataSize=validNum)
    testdataset = tfrecord_read.getShards(testFiles, batchSize=batchSize,imgDims=imgDims,  dataSize=testNum, test=True)

    if decaySchedule != None:
       print('Decaying learning rate using {}'.format(decaySchedule))
       decay = 0
       scheduleKwargs = params['optimizer']['decayparams']
       scheduler = getattr(decay_schedules, decaySchedule)(**scheduleKwargs)
       callbacks += [LearningRateScheduler(scheduler)]
       scheduler.plot(epoch, modelName + ': ' + decaySchedule + ' learning rate')
       plt.savefig(os.path.join(outPath,modelName+'.png'))

    #tensorBoard = tf.keras.callbacks.TensorBoard(log_dir='tensorboard/logs')
    #checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(checkpointPath,modelName), monitor=metric, period=10)
    #callbacks = [checkpoint]


    devices = ['/device:GPU:{}'.format(i) for i in range(multiDict['num'])]
    strategy = tf.distribute.MirroredStrategy(devices)

    with strategy.scope():

        print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        if loss=='weightedBinaryCrossEntropy':
            print('Loss: Weighted binaryCrossEntropy')
            posWeight = float(weights[0])
            lossObject = WeightedBinaryCrossEntropy(posWeight)
        elif loss=='diceloss':
            print('Loss: diceloss')
            lossObject = DiceLoss()
        else:
            raise ValueError('No loss requested, please update config file')

        if optimizer=='adam':
            print('Optimizer: {}'.format(optimizer))
            optimizer = keras.optimizers.Adam(**optKwargs)
        elif optimizer=='Nadam':
            print('Optimizer: {}'.format(optimizer))
            optimizer = keras.optimizers.NAdam(**optKwargs)
        elif optimizer=='SGD':
            print('Optimizer: {}'.format(optimizer))
            optimizer=keras.optimizers.SGD(**optKwargs)
        else:
            raise ValueError('No optimizer selected, please update config file')

        if model == 'fcn8':
            print('Model: {}'.format(model))
            with tf.device('/cpu:0'):
                if api == 'functional':
                    fcn = FCN()
                    model = fcn.getFCN8()
                elif api=='subclass':
                    model = FCN()

        elif model == 'unet':
            print('Model: {}'.format(model))
            with tf.device('/cpu:0'):
                if api=='functional':
                    unetModel = unet2.UnetFunc()
                    model = unetModel.unet()
                elif api=='subclass':
                    model = unetsc.UnetSC(filters=filters)
                    model.build((1, imgDims, imgDims, 3))

        elif model == 'unetmini':
            print('Model: {}'.format(model))
            with tf.device('/cpu:0'):
                if api == 'functional':
                    unetminiModel = UnetMini(filters=filters)
                    model = unetminiModel.unetmini()
                elif api=='subclass':
                    model = UnetMini(filters)

        elif model == 'resunet':
            print('Model: {}'.format(model))
            with tf.device('/cpu:0'):
                if api=='functional':
                    resunetModel =  ResUnet(filters)
                    model = resunetModel.ResUnetFunc()
                elif api=='subclass':
                    model = ResunetSc(filters)

        elif model == 'resunet-a':
            print('Model: {}'.format(model))
            with tf.device('/cpu:0'):
                if api=='functional':
                    resunetModel =  ResUnetA(filters)
                    model = resunetModel.ResUnetAFunc()
                elif api=='subclass':
                    model = ResunetASc(filters)

        elif model == 'attention':
            print('Model: {}'.format(model))
            with tf.device('/cpu:0'):
                if api == 'functional':
                    attenModel = AttenUnetFunc(filters)
                    model = attenModel.attenUnet()
                elif api=='subclass':
                    model = AttenUnetSC(filters)
        else:
            raise ValueError('No model requested, please update config file')

        
        if trainapi == 'custom':

            trainDistDataset = strategy.experimental_distribute_dataset(trainDataset)
            validDistDataset = strategy.experimental_distribute_dataset(validDataset)

            train = distributed_train.DistributeTrain(epoch, model, optimizer, lossObject, batchSize, strategy, trainSteps, testSteps, imgDims)
            model, history = train.forward(trainDistDataset, validDistDataset)

            model.save(os.path.join('/home/verghese/models',str(currentDate), modelName + '_' + currentTime.strftime('%H:%M')), save_format='tf')
            #with open(os.path.join('home/verghese/models/', currentDate, currentTime, modelName+'_history'), 'wb') as f:
         #      pickle.dump(history, f)

        elif trainapi == 'fit':
            #ToDo: does .fit handle strategy.experimentatl_distribute_dataset
            model.compile(optimizer=optimizer, loss=losObject, metrics=[diceCoef])
            history = model.fit(x=trainDistDataset, steps_per_epoch=trainSteps, validation_data=validDistDataset, validation_steps=validSteps, epochs=epoch) 
            
            model.save(os.path.join('/home/verghese/models', str(currentDate), modelName + '_' +currentTime.strftime('%H:%M')+'.h5')
            #with open(os.path.join('home/verghese/models', str(currentDate), modelName + '_'+currentTime.strftime('%H:%M')+'.h5','wb') as f:
                      #pickle.dump(history, f)


    patchpredict = PatchPredictions(model, modelName, batchSize)
    patchpredict(testDataset, outPath)
    wsipredict = WSIPredictions(model, modelName, feature, magnification, imgDims)
    wsipredict(recordsPath)

    getTrainMetric(history, 'loss', 'valloss', outPath, modelName)
    getTrainMetric(history, metric, 'valmetric', outPath, modelName)

    return result



if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-rp', '--recordpath', required=True, help='path to tfrecords')
    ap.add_argument('-op', '--outpath', required=True, help='output path for predictions')
    ap.add_argument('-cp', '--checkpointpath', required=True, help='path for checkpoint files')
    ap.add_argument('-cf', '--configfile', help='file containing parameters')

    args = vars(ap.parse_args())

    trainSegmentationModel(args)
