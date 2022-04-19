import os

import tensorflow as tf
import seaborn as sns
import numpy as np
import operator
import matplotlib.pyplot as plt

def oneHotToMask(onehot):

    nClasses =  onehot.shape[-1]
    idx = tf.argmax(onehot, axis=-1)
    colors = sns.color_palette('hls', nClasses)
    multimask = tf.gather(colors, idx)
    multimask = np.where(multimask[:,:,:]==colors[0], 0, multimask[:,:,:])

    return multimask


def resizeImage(dim, factor=2048, threshold=0, op=operator.gt):
    boundaries = [factor*i for i in range(100000)]
    boundaries = [f for f in boundaries if op(f,threshold)]
    diff = list(map(lambda x: abs(dim-x), boundaries))
    newDim = boundaries[diff.index(min(diff))]

    return newDim



def getTrainCurves(history, tMetricName, vMetricName, outPath, modelname):
    sns.set_style('dark')
    trainMetric = history[tMetricName]
    valMetric = history[vMetricName]
    epochs = range(len(trainMetric))    
    fig = plt.figure(figsize=(8,5))
    sns.lineplot(range(len(trainMetric)),trainMetric,markers=True,dashes=False,label='Training'+tMetricName)
    sns.lineplot(range(len(trainMetric)),valMetric,markers=True,dashes=False,label='Validation'+tMetricName)
    plt.title('Training and validation'+tMetricName)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    fig.savefig(os.path.join(outPath,modelname+tMetricName+'_graph.png'))
    plt.close()

def getFiles(filesPath, ext):
    filesLst=[]
    for path, subdirs, files in os.walk(filesPath):
        for name in files:
            if name.endswith(ext):
                filesLst.append(os.path.join(path,name))
    return filesLst
