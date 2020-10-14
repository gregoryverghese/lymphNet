import json
import os
import argparse
import pandas as pd
from ln_segmentation_2 import trainSegmentationModel


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
    losses = ['weightedBinaryCrossEntropy']
    dropouts = [0]
    #augmentation = [["Flip", "Rotate90", "Color"], ["Flip","Rotate90"]]
    augmentation = [[], ["Flip","Rotate90"], ["Flip", "Rotate90","Rotate"],["Flip","Rotate90","Color"],["Color", "Rotate90", "Flip", "Rotate"]]
    #augmentation = [[]]
   
    for a in augmentation:
       for l in losses:
           for d in dropouts:

              with open(configTemplate) as jsonFile:
                  jsonDict = json.load(jsonFile)

              jsonDict['dropout'] = d
              jsonDict['loss'] = l
              jsonDict['augmentation']['methods'] = a
              name = jsonDict['modelname']
              name = name.replace('$loss', l)
              name = name.replace('$drop', str(d))

              augInitials = [i[0] for i in a]
              name = name.replace('$augment', ''.join(augInitials))
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
