import tensorflow as tf
import seaborn as sns
import numpy as np

def oneHotToMask(onehot):

    nClasses =  onehot.shape[-1]
    idx = tf.argmax(onehot, axis=-1)
    Colors = sns.color_palette('hls', nClasses)
    multimask = tf.gather(colors, idx)

    return multimask


def resizeImage(dim, factor=2048):

    boundaries = [factor*i for i in range(100)]
    diff = list(map(lambda x: abs(dim-x), boundaries))
    newDim = boundaries[diff.index(min(diff))]

    return newDim


