import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=K.epsilon(), reduction=tf.keras.losses.Reduction.NONE):
        super(DiceLoss, self).__init__(reduction=reduction)
        self.smooth = smooth

    def call(self, yPred, yTrue):

        yPred = tf.cast(yPred, tf.float32)
        yTrue = tf.cast(yTrue, tf.float32)

        intersection = K.sum(yTrue * yPred, axis=[1, 2])
        union = K.sum(yTrue, axis=[1, 2]) + K.sum(yPred, axis=[1, 2])
        dice = K.mean((2. * intersection + K.epsilon())/(union + K.epsilon()), axis=0)
        print(dice)
        return 1 - dice


'''
class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, posWeight, reduction=tf.keras.losses.Reduction.NONE):
        super().__init__(reduction=reduction)
        self.posWeight = posWeight

    def call(self, yTrue, yPred):

        yPred = tf.cast(yPred, tf.float32)
        yTrue = tf.cast(yTrue, tf.float32)
        tf.math.log(yPred/(1-yPred))
        yPred = tf.clip_by_value(yPred, K.epsilon(), 1-K.epsilon())
        logits = tf.math.log(yPred/(1-yPred))
        return tf.nn.weighted_cross_entropy_with_logits(logits=logits, labels=yTrue, pos_weight=self.posWeight)


'''
class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, pos_weight, weight=1, from_logits=False,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name='weighted_binary_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight
        self.weight = 1
        self.from_logits = from_logits

    def call(self, y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        ce = tf.losses.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits)[:,None]
        ce = self.weight * (ce*(1-y_true) + self.pos_weight*ce*(y_true))
        return ce
