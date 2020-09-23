import tensorflow as tf
import seaborn as sns
import numpy as np

def oneHotToMask(onehot):

    nClasses =  onehot.shape[-1]
    idx = tf.argmax(onehot, axis=-1)
    Colors = sns.color_palette('hls', nClasses)
    multimask = tf.gather(colors, idx)

    return multimask


