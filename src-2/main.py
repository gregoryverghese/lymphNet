'''
main.py: main module that sets up analysis, models
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

from distributed_train import DistributedTraining 
from models import fcn8,unet,mobile,resunet,resunet_a,unet_mini,atten_unet
from models import multiscale,multi_atten,deeplabv3
from models import deeplabv3
from data.tfrecord_read import TFRecordLoader
from utilities import decay_schedules
from utilities.evaluation import diceCoef
from predict import WSIPredictions, PatchPredictions
from utilities.utils import getTrainCurves
from utilities.custom_loss_classes import BinaryXEntropy, DiceLoss, CategoricalXEntropy

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


#LOSSFUNCTIONS={
#              'binary-xentropy':BinaryCrossEntropy
#              'categorical-xentropy':CategoricalCrossEntropy,
#              'diceloss':DiceLoss
#              }


def data_loader(path,config):
    
    #load training files
    train_path = os.path.join(path,'train','*.tfrecords')
    train_files = glob.glob(train_path)
    print('crazy')
    train_loader=TFRecordLoader(train_files,
                                'train',
                                config['imageDims'],
                                config['tasktype'],
                                config['batchSize'])
    train_loader.record_size()
    print(f'n={train_loader.tile_nums}')
    
    #augment
    aug_methods=config['augmentation']['methods']
    aug_parameters=config['augmentation']['parameters']
    train_loader.load(config['batchSize'])
    train_loader.augment(aug_methods,aug_parameters)

    #normalize
    norm_methods=config['normalize']['methods']
    norm_parameters=config['normalize']['parameters']
    train_loader.normalize(norm_methods,norm_parameters)
    
    #load validation files
    valid_path = os.path.join(path,'validation','*.tfrecords')
    valid_files = glob.glob(valid_path)
    valid_loader=TFRecordLoader(valid_files,
                               'valid',
                               config['imageDims'],
                               config['tasktype'],
                               config['batchSize'])

    valid_loader.record_size()
    print(f'n={valid_loader.tile_nums}')
    valid_loader.load(1)
    return train_loader,valid_loader


def main(args,name):
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

    with open(args.config_file) as json_file:
        config = json.load(json_file)

    path = os.path.join(args.record_path,args.record_dir)
    train_loader,valid_loader=data_loader(path,config)
        
    #get devices
    devices = tf.config.experimental.list_physical_devices('GPU')
    devices = [x.name.replace('/physical_device:', '') for x in devices] 
    #devices = ['/device:GPU:{}'.format(i) for i in range(multiDict['num'])]

    nnParams={'filters':config['model']['parameters']['filters'],
              'finalActivation':config['model']['parameters']['finalActivation'],
              'nOutput':config['nClasses'],
              #'dims':config['imageDims'],
              'upTypeName':config['upTypeName']
                }
    #modelname='attention'
    strategy = tf.distribute.MirroredStrategy(devices)
    with strategy.scope():
        boundaries=[30, 60]
        values=[0.001,0.0005,0.0001]
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries,values)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        criterion = BinaryXEntropy(config['weights'][0])
        with tf.device('/cpu:0'):
                model=FUNCMODELS[args.model_name](**nnParams)
                model=model.build()
        #table = PrettyTable(['Model', 'Loss', 'Optimizer', 'Devices','DecaySchedule'])
        #table.add_row([modelname, loss, optimizername, len(devices),decaySchedule['keras']])
        #print(table)
        #call distributed training script to allow for training on multiple gpus
        train_dataset = strategy.experimental_distribute_dataset(train_loader.dataset)
        valid_dataset = strategy.experimental_distribute_dataset(valid_loader.dataset)
        train = DistributedTraining(model,
                                    train_loader,
                                    valid_loader,
                                    optimizer, 
                                    criterion,
                                    config['batchSize'],
                                    config['epoch'],
                                    strategy, 
                                    config['imageDims'], 
                                    config['stopthresholds'],
                                    config['activationthreshold'],
                                    config['modelname'],
                                    config['tasktype'])

        model, history = train.forward()

        #model.save(os.path.join(outModelPath, modelName + '_' + currentTime + '.h5'))
    #with open(os.path.join(outModelPath,  modelName+'_'+ currentTime + '_history'), 'wb') as f:
        #pickle.dump(history, f)

    #call predict on individual patches and entire annotated wsi region
    #patchpredict = PatchPredictions(model, modelName, batchSize, currentTime, currentDate, activationthreshold)
    #patchpredict(testdataset, os.path.join(outPath, 'predictions'))
    
    #outpath='/home/verghese/breastcancer_ln_deeplearning/output/predictions/wsi'
    #wsipredict = WSIPredictions(model, 
                                #modelName, 
                                #feature,
                                #magnification,
                                #step, 
                                #step,
                                #activationthreshold, 
                                #currentTime,
                                #currentDate, 
                                #tasktype, 
                                #normalize,
                                #normalizeParams)

    #result = wsipredict(os.path.join(recordsPath, 'tfrecords_wsi'), outPath)
    #ToDo: replace generic trainmetric with metric name from config file
    #getTrainCurves(history,'trainloss','valloss',outCurvePath,modelName+'_'+currentTime)
    #getTrainCurves(history,'trainmetric', 'valmetric', outCurvePath,modelName+'_'+currentTime)

    #finally save the config for this file with the model and predictions
    #configPath=os.path.join(outModelPath,modelName+'_'+currentTime+'_config.json')
    #with open(configPath, 'w') as jsonFile:
    #json.dump(params, jsonFile)

    #return result
"""

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-rp', '--record_path', required=True, help='path to tfrecords')
    ap.add_argument('-rd', '--record_dir', required=True, help='directory for specific tfrecords dataset')    
    ap.add_argument('-op', '--save_path', required=True, help='path to save output')
    ap.add_argument('-cp', '--checkpoint_path', required=True, help='path for checkpoint files')
    ap.add_argument('-cf', '--config_file', help='config file with parameters')
    ap.add_argument('-mn', '--model_name', help='neural network model')
    args = ap.parse_args()
  

    print(args)
    #get current date and time for model name
    curr_date=str(datetime.date.today())
    curr_time=datetime.datetime.now().strftime('%H:%M')
    name=curr_date+'_'+curr_time

    #set up paths for models, training curves and predictions
    save_path = os.path.join(args.save_path,curr_date+'_'+curr_time)
    os.makedirs(save_path,exist_ok=True)

    save_model_path = os.path.join(save_path,'models')
    os.makedirs(save_model_path,exist_ok=True)

    out_curve_path = os.path.join(save_path,'curves')
    os.makedirs(out_curve_path,exist_ok=True)

    out_predict_path=os.path.join(save_path,'predictions')
    os.makedirs(out_predict_path,exist_ok=True)

    main(args,name)
"""
