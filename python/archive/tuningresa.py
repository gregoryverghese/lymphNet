import json
import os
import argparse
import pandas as pd
from ln_segmentation import trainSegmentationModel


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
    losses = ['weightedBinaryCrossEntropy', 'diceloss']
    dropouts = [0]
    augmentation = [["Flip","Rotate90"], ["Flip", "Rotate90", "Color"]]
    models = ["resunet-a"]
    i=0
    for m in models:
        print(m, flush=True)
        for a in augmentation:
            for l in losses:
                for d in dropouts:
                    i=i+1 
                    if i>0:
                        print(i, flush=True)
                        print('loss: {} \n dropout: {} \n model: {} \n augmentation: {}'.format(l, d, m, a), flush=True)

                        with open(configTemplate) as jsonFile:
                            jsonDict = json.load(jsonFile)

                        jsonDict['dropout'] = d
                        jsonDict['loss'] = l
                        jsonDict['model'] = m
                        jsonDict['augmentation'] = a
                        name = jsonDict['modelname']
                        name = name.replace('$loss', l)
                        name = name.replace('$drop', str(d))
                        name = name.replace('$model',m)
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
