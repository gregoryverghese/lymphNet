#!/usr/bin/env python3

'''
custom_classes.py: script contains classes inheriting from tf.keras.losses.loss
'''

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=K.epsilon(), reduction=tf.keras.losses.Reduction.NONE):
        super(DiceLoss, self).__init__(reduction=reduction)
        self.smooth = smooth

    def call(self, yTrue, yPred):
        yPred = tf.cast(yPred, tf.float32)
        yTrue = tf.cast(yTrue, tf.float32)
        intersection = K.sum(yTrue * yPred, axis=[1, 2, 3])
        union = K.sum(yTrue, axis=[1, 2, 3]) + K.sum(yPred, axis=[1, 2, 3])
        dice = K.mean((2. * intersection + K.epsilon())/(union + K.epsilon()), axis=0)
        return 1 - dice


class BinaryXEntropy(tf.keras.losses.Loss):
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


class CategoricalXEntropy(tf.keras.losses.Loss):
    def __init__(self, weights, reduction=tf.keras.losses.Reduction.NONE):
        super().__init__(reduction=reduction)
        self.weights = weights

    def call(self, yTrue, yPred):
        yPred = tf.cast(yPred, tf.float32)
        yTrue = tf.cast(yTrue, tf.float32)
        pixelXEntropies = yTrue * (tf.math.log(yPred))
        weightXEntropies = self.weights * pixelXEntropies
        return -tf.reduce_sum(weightXEntropies, axis=-1)



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


class WeightedCategoricalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, weights, reduction=tf.keras.losses.Reduction.NONE):
        super().__init__(reduction=reduction)
        self.weights = weights

    def call(self, yTrue, yPred):

        yPred = tf.cast(yPred, tf.float32)
        yTrue = tf.cast(yTrue, tf.float32)

        pixelXEntropies = yTrue * (tf.math.log(yPred))
        weightXEntropies = self.weights * pixelXEntropies

        return -tf.reduce_sum(weightXEntropies, axis=-1)
        
class generalized_dice_loss(tf.keras.losses.Loss):
    def __init__(self, weights, reduction=tf.keras.losses.Reduction.NONE):
        super().__init__(reduction=reduction)
        self.weights = weights

    def call(self,yTrue,yPred):
        yPred = tf.cast(yPred,tf.float32)
        yTrue = tf.cast(yTrue,tf.float32)
        def generalized_dice_coeff(y_true, y_pred):
            Ncl = y_pred.shape[-1]
            #w = K.zeros(shape=(Ncl,))
            w = K.sum(y_true, axis=(0,1,2))
            w = 1/(w**2+0.000001)
            # Compute gen dice coef:
            numerator = y_true*y_pred
            numerator = w*K.sum(numerator,(0,1,2,3))
            numerator = K.sum(numerator)
            denominator = y_true+y_pred
            denominator = w*K.sum(denominator,(0,1,2,3))
            denominator = K.sum(denominator)
            gen_dice_coef = 2*numerator/denominator
            return gen_dice_coef
        return 1 - generalized_dice_coeff(yTrue, yPred)
        
        
class WCE_and_dice_loss(tf.keras.losses.Loss):
    def __init__(self,weights,reduction=tf.keras.losses.Reduction.NONE):
        super().__init__(reduction=reduction)
        self.weights = weights
    def call(self,yTrue,yPred):
        yPred = tf.cast(yPred,tf.float32)
        yTrue = tf.cast(yTrue,tf.float32)
        def dice_coef(y_True, y_Pred, smooth=1):
            intersection = K.sum(y_True * y_Pred, axis=[1,2,3]) ##y_true and y_pred are both matrix（Unet）
            union = K.sum(y_True, axis=[1,2,3]) + K.sum(y_Pred, axis=[1,2,3])
            return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        def dice_p_bce(in_gt, in_pred):
            return 0.5*binary_crossentropy(in_gt, in_pred) - 0.5*dice_coef(in_gt, in_pred)
        return dice_p_bce(yTrue,yPred)


class Focal_loss(tf.keras.losses.Loss):
    def __init__(self,reduction=tf.keras.losses.Reduction.NONE):
        super().__init__(reduction=reduction)
    def call(self,yTrue,yPred):
        yPred = tf.cast(yPred,tf.float32)
        yTrue = tf.cast(yTrue,tf.float32)

        def focal_loss(yPred, yTrue, alpha=0.4, gamma=2):

            zeros = tf.zeros_like(yPred, dtype=yPred.dtype)
            pos_p_sub = tf.where(yTrue > zeros, yTrue - yPred, zeros) 

            neg_p_sub = tf.where(yTrue > zeros, zeros, yPred)
            per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.math.log(tf.clip_by_value(yPred, 1e-8, 1.0)) \
                                  - (1 - alpha) * (neg_p_sub ** gamma) * tf.math.log(tf.clip_by_value(1.0 - yPred, 1e-8, 1.0))

            return tf.reduce_sum(per_entry_cross_ent)
        return focal_loss(yPred,yTrue)

class Tversky_loss(tf.keras.losses.Loss):
    def __init__(self,weights,reduction=tf.keras.losses.Reduction.NONE):
        super().__init__(reduction=reduction)
    def call(self,yTrue,yPred):
        yPred = tf.cast(yPred,tf.float32)
        yTrue = tf.cast(yTrue,tf.float32)
        def tversky(y_true, y_pred,smooth=1):
            y_true_pos = K.flatten(y_true)
            y_pred_pos = K.flatten(y_pred)
            true_pos = K.sum(y_true_pos * y_pred_pos)
            false_neg = K.sum(y_true_pos * (1-y_pred_pos))
            false_pos = K.sum((1-y_true_pos)*y_pred_pos)
            alpha = 0.7
            return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
        def tversky_loss(y_true, y_pred):
            print(tversky_loss(yTrue,yPred))
            return 1 - tversky(y_true,y_pred)
            #print(tversky_loss(yTrue,yPred))
        return tversky_loss(yTrue,yPred)


class Switching_loss(tf.keras.losses.Loss):
    def __init__(self,posWeight,reduction=tf.keras.losses.Reduction.NONE):
        super().__init__(reduction=reduction)
#        self.weights = weights
        self.posWeight = posWeight
    def call(self,yTrue,yPred):
        yPred = tf.cast(yPred,tf.float32)
        yTrue = tf.cast(yTrue,tf.float32)
        cn=tf.reduce_sum(K.sum(yTrue,axis=[1,2,3]))
        dim_yTrue=yTrue.get_shape().as_list()
        ct=int(dim_yTrue[0])*int(dim_yTrue[1])*int(dim_yTrue[2])*int(dim_yTrue[3])
        tau1=cn/ct
        def dice_loss(yTrue,yPred):
            intersection = K.sum(yTrue * yPred, axis=[1, 2, 3])
            union = K.sum(yTrue, axis=[1, 2, 3]) + K.sum(yPred, axis=[1, 2, 3])
            dice = K.mean((2. * intersection + K.epsilon())/(union + K.epsilon()), axis=0)
            return 1 - dice
        def inverted_dice_loss(yTrue,yPred):
            intersection = K.sum((1-yTrue) * (1-yPred), axis=[1, 2, 3])
            union = K.sum((1-yTrue), axis=[1, 2, 3]) + K.sum((1-yPred), axis=[1, 2, 3])
            dice = K.mean((2. * intersection + K.epsilon())/(union + K.epsilon()), axis=0)
            return 1 - dice
        def BCE(yTrue,yPred):
            yPred = tf.clip_by_value(yPred, K.epsilon(), 1-K.epsilon())
            logits = tf.math.log(yPred/(1-yPred))
            bce=tf.nn.weighted_cross_entropy_with_logits(logits=logits, labels=yTrue, pos_weight=self.posWeight)
            #bce=0-K.sum(tf.math.log(yPred+K.epsilon()),axis=[1,2,3])
            return bce
        def switching_loss(yTrue,yPred,lambda1=0.75,tau=0.7):
            #only consider the target is small.
            sys.stderr.write("dice_score")
            #print("DICE: "+str(tf.print(dice_loss(yTrue,yPred),output_stream=sys.stderr)))
            sys.stderr.write("inverted_dice_score")
            #print("INVERTED_DICE: "+str(tf.print(inverted_dice_loss(yTrue,yPred),output_stream=sys.stderr)))
            sys.stderr.write("BCE")
           # print("BCE: "+str(tf.print(BCE(yTrue,yPred),output_stream=sys.stderr)))
            #print(tau1/tau)
            SL=BCE(yTrue,yPred)
            if(tau1>=tau):
                #print("condition 1!!!!!!!!!!!!!!!")
                SL=BCE(yTrue,yPred)+lambda1*dice_loss(yTrue,yPred)+(1-lambda1)*inverted_dice_loss(yTrue,yPred)
            if(tau1<tau):
                #print("condition 2!!!!!!!!!!!!!!!!!!!!")
                SL=BCE(yTrue,yPred)+(1-lambda1)*dice_loss(yTrue,yPred)+lambda1*inverted_dice_loss(yTrue,yPred)
            return SL
        return switching_loss(yTrue,yPred)







def get_criterion():
    if x=='cross_entropy':
        criterion=BinaryXEntropy
    elif x=='dice_loss':
        criterion=BinaryXEntropy
    elif x=='weighted_binary_cross_entropy':
        criterion=WeightedBinaryCrossEntropy
    elif x=='weighted_categorical_cross_entropy':
        criterion=WeightedCategoricalCrossEntropy
    elif x=='generalized_dice_loss':
        criterion=generalized_dice_loss
    elif x=='wce_and_dice_loss':
        criterion=WCE_and_dice_loss
    elif x=='focal_loss':
        criterion=Focal_loss
    elif x=='tversky_loss':
        criterion=Tversky_loss
    elif x=='switching_loss':
        criterion=Switching_loss      
    return criterion


































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

        print('shapessss', K.int_shape(y_pred))

        ce = tf.losses.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits)[:,None]
        ce = self.weight * (ce*(1-y_true) + self.pos_weight*ce*(y_true))
        return ce
    '''
