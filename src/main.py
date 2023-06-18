'''
main.py: main script sets up analysis, data and models. Calls

1. distributed training
2. prediction
'''

import os
import sys
import csv
import glob
import pickle
import random
import argparse
import yaml
import datetime

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = 2

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
#from utilities import decay_schedules
from utilities.evaluation import diceCoef
from predict import test_predictions
from utilities.utils import get_train_curves, save_experiment
from utilities.custom_loss_classes import BinaryXEntropy, DiceLoss, CategoricalXEntropy

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FUNCMODELS={
    'unet':unet.Unet,
    'unetmini':unet_mini.UnetMini,
    'attention':atten_unet.AttenUnet,
    'multiscale':multiscale.MSUnet,
    'multiatten':multi_atten.MultiAtten,
    'resunet':resunet.ResUnet,
    'fcn8':unet.Unet,
    'mobile':mobile.MobileUnet,
    'deeplabv3plus':deeplabv3.DeepLabV3Plus
        }


LOSSFUNCTIONS={
    'wCE':BinaryXEntropy,
    'wCCE':CategoricalXEntropy,
    'DL':DiceLoss
              }


def data_loader(path,config):
    
    #load training files
    print('Loading training data')
    train_path = os.path.join(path,'train','*.tfrecords')
    train_files = glob.glob(train_path)
    train_loader=TFRecordLoader(train_files,
                                'Train',
                                config['image_dims'],
                                config['task_type'],
                                int(config['batch_size']))
    train_loader.record_size()
    print(f'tiles: n={train_loader.tile_nums}; steps:n={train_loader.steps}')
    
    #augmention
    aug_methods=config['augmentation']['methods']
    aug_parameters=config['augmentation']
    
    train_loader.load(int(config['batch_size']))
    train_loader.augment(aug_methods,aug_parameters)

    #normalize
    norm_methods=config['normalize']['methods']
    norm_parameters=config['normalize']
    train_loader.normalize(norm_methods,norm_parameters)
    
    print('Loading validation data')
    #load validation files
    valid_path = os.path.join(path,'validation','*.tfrecords')
    valid_files = glob.glob(valid_path)
    valid_loader=TFRecordLoader(valid_files,
                               'test',
                               config['image_dims'],
                               config['task_type'],
                               int(config['batch_size'])
                               )

    valid_loader.record_size()
    print(f'tiles: n={valid_loader.tile_nums}; steps:n={valid_loader.steps}')
    valid_loader.load(int(config['batch_size']))
    valid_loader.normalize(norm_methods,norm_parameters)

    return train_loader,valid_loader


def main(args,config,name,save_path):
    '''
    sets up analysis based on config file. Following design choices:

    1. model
    2. optimizer and optimizer decay
    3. loss function

    load tfrecords data, calls custom training and prediction scripts. 
    Parameters and model are saved down
    
    :param args: command line arguments
    :param config: config file (yaml)
    :param name: experiment name
    :param save_path: path for experiment output
    :returns result: avg dice and iou score
    '''
    #tensorflow logs
    train_log_dir = os.path.join(save_path,'tensorboard_logs', 'train')
    test_log_dir = os.path.join(save_path, 'tensorboard_logs', 'test') 
    train_writer = tf.summary.create_file_writer(train_log_dir)
    test_writer = tf.summary.create_file_writer(test_log_dir)

    #set up train and valid loaders
    data_path = os.path.join(args.record_path,args.record_dir)
    train_loader,valid_loader=data_loader(data_path,config)
        
    #collect gpus
    devices = tf.config.experimental.list_physical_devices('GPU')
    devices = [x.name.replace('/physical_device:', '') for x in devices]
    print(devices)
    #devices = ['/device:GPU:{}'.format(i) for i in range(multiDict['num'])]
    
    #set up model parameters
    model_params={
        'filters':config['model']['filters'],
        'final_activation':config['model']['final_activation'],
        'dropout':config['model']['dropout'],
        'n_output':config['num_classes'],
            }
    loss_params={'weights':config['weights'][0]}

    #use distributed training (multi-gpu training)
    strategy = tf.distribute.MirroredStrategy(devices)
    with strategy.scope():
        #boundaries=[30, 60]
        #values=[0.001,0.0005,0.0001]
        #lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries,values)
        #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        #criterion = LOSSFUNCTIONS[config['loss'][0]](**loss_params)
        criterion = BinaryXEntropy(config['weights'][0])
        #with tf.device('/cpu:0'):
        model=FUNCMODELS[args.model_name](**model_params)
        model=model.build()
 
    train_dataset_dist = strategy.experimental_distribute_dataset(train_loader.dataset)
    valid_dataset_dist = strategy.experimental_distribute_dataset(valid_loader.dataset)
    train_loader.dataset = train_dataset_dist
    valid_loader.dataset = valid_dataset_dist

    train = DistributedTraining(
        model,
        train_loader,
        valid_loader,
        optimizer, 
        criterion,
        strategy, 
        config['batch_size'],
        config['epochs'],
        config['image_dims'], 
        config['early_stopping'],
        config['threshold'],
        config['task_type'],
        train_writer,
        test_writer,
        config,
        save_path)
    
    model, history = train.forward()
    #save model, config and training curves
    #model_save_path=os.path.join(save_path,'models')
    #save_experiment(model,config,history,name,model_save_path)
    curve_save_path=os.path.join(save_path,'curves')
    get_train_curves(history,'train_loss','val_loss',curve_save_path)
    get_train_curves(history,'train_metric', 'val_metric',curve_save_path)
    pre=False

    if pre:
        print('prediction')
        result=test_predictions(
            model,
            args.test_path,
            args.save_path,
            config['feature'],
            config['threshold'],
            config['step'],
            config['normalize']['methods'],
            config['normalize']['channel_mean'],
            config['normalize']['channel_std']
        )
    #return result


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-rp', '--record_path', required=True, help='path to tfrecords')
    ap.add_argument('-rd', '--record_dir', required=True, help='directory for specific tfrecords dataset')    
    ap.add_argument('-op', '--save_path', required=True, help='path to save output')
    ap.add_argument('-tp', '--test_path', required=True, help='path to test images')
    ap.add_argument('-cp', '--checkpoint_path', required=True, help='path for checkpoint files')
    ap.add_argument('-cf', '--config_file', help='config file with parameters')
    ap.add_argument('-mn', '--model_name', help='neural network model')
    ap.add_argument('-p', '--predict', help='set this flag to run the trained model on test set automatically')
    args = ap.parse_args()
    
    #get current date and time for model name
    curr_date=str(datetime.date.today())
    curr_time=datetime.datetime.now().strftime('%H:%M')

    with open(args.config_file) as yaml_file:
        config=yaml.load(yaml_file, Loader=yaml.FullLoader)

    name=config['name']+'_'+curr_date+'_'+curr_time
    print(name)
    #set up paths for models, training curves and predictions
    save_path = os.path.join(args.save_path,name)
    os.makedirs(save_path,exist_ok=True)

    save_model_path = os.path.join(save_path,'models')
    os.makedirs(save_model_path,exist_ok=True)

    save_curve_path = os.path.join(save_path,'curves')
    os.makedirs(save_curve_path,exist_ok=True)

    save_predict_path=os.path.join(save_path,'predictions')
    os.makedirs(save_predict_path,exist_ok=True)

    save_logs_path=os.path.join(save_path,'tensorflow-logs')
    os.makedirs(save_logs_path,exist_ok=True)

    main(args,config,name,save_path)

