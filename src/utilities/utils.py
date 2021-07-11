import tensorflow as tf
import seaborn as sns
import numpy as np
import operator


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


def mask2rgb(mask):
    n_classes=len(np.unique(mask))
    colors=sns.color_palette('hls',n_classes)
    rgb_mask=np.zeros(mask.shape+(3,))
    for c in range(1,n_classes):
        t=(mask==c)
        rgb_mask[:,:,0][t]=colors[c][0]
        rgb_mask[:,:,1][t]=colors[c][1]
        rgb_mask[:,:,2][t]=colors[c][2]
    return rgb_mask


