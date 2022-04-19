#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
augmentations_tuning.py: opens config fie and iterates
over different combinations of values for the augmentation
techniques for the purpose of hyperparameter tuning.
'''

import json
import os
import argparsei

import pandas as pd
from ln_segmentation import trainSegmentationModel

__author__ = 'Gregory Verghese'
__email__ = 'gregory.verghese@kcl.ac.uk'


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-rp', '--recordpath', required=True, help='path to tfrecords')
    ap.add_argument('-rd', '--recordDir', required=True, help='directory for the tfrecords dataset')
    ap.add_argument('-op', '--outpath', required=True, help='output path for predictions')
    ap.add_argument('-cp', '--checkpointpath', required=True, help='path for checkpoint files')
    ap.add_argument('-cf', '--configfile', help='file containing parameters')
    
    args = vars(ap.parse_args())

    indexes = []
    results = []
    configTemplate = args['configfile']

    print('config file', configTemplate)

    with open(configTemplate) as jsonFile:
        jsonDict = json.load(jsonFile)

    analysisName = jsonDict['modelname']

    hue = [0.05,0.15,0.25,0.35]
    saturation = [[0.1,1], [0.1,3], [0.1,5]]
    brightness = [0.1,0.2,0.3]
    contrast = [[0.8,1.2],[0.8,1.4]]
    
    for h in hue:
        for s in saturation:
            for b in brightness:
                for c in contrast:
                    print('hue: {} \n saturation: {} \n brightness: {} \n contrast: {}'.format(h, s, b, c), flush=True)

                    with open(configTemplate) as jsonFile:
                        jsonDict = json.load(jsonFile)

                    name = jsonDict['modelname']
                    name = name.replace('$hue', 'hue='+str(h))
                    name = name.replace('$saturation', 'satur='+str(s))
                    name = name.replace('$bright', 'bright='+str(b))
                    name = name.replace('$contrast', 'contrast='+str(c))

                    name = name.replace('$model','model='+jsonDict['model']['network'])
                    name = name.replace('$loss', 'loss='+jsonDict['loss'])
                    name = name.replace('$batch', 'batch='+str(jsonDict['batchSize']))

                    jsonDict['augmentation']['parameters']['contrast']=c
                    jsonDict['augmentation']['parameters']['brightness']=b
                    jsonDict['augmentation']['parameters']['saturation']=s
                    jsonDict['augmentation']['parameters']['hue']=h
                    
                    print('Now executing following model: {}'.format(name))

                    jsonDict['modelname'] = name
                    configFile = '/home/verghese/config/' + name+'.json'
                    args['configfile'] = configFile

                    with open(configFile, 'w') as jsonFile:
                        json.dump(jsonDict, jsonFile)

                    result = trainSegmentationModel(args)
                    indexes.append(name)
                    results.append(result)

    df = pd.DataFrame({'dice': results}, index=indexes)
    df.to_csv(analysisName+'.csv')
