import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import cv2
import glob
import os
import numpy as np
from scripts.python.utilities.evaluation import diceCoef, iouScore


class WSIPrediction():
    '''
    class that applies model to predict 
    on large region/entire whole slide image
    '''
    def __init__(self,model,image,mask,step):

        self.model=model
        self.image=image
        self.mask=mask
        self._prediction=None
        self.step=step

    @property
    def diceScore(self):
        return diceCoef(self.prediction, self.mask)

    @property
    def iouScore(self):
        return iouScore(self.prediction,self.mask)


    def split(self,xStep,yStep):
        _,x,y,_=K.int_shape(self.image)
        for j in range(0,y,yStep):
            for i in range(0,x,xStep):
                yield self.image[:,i:i+xStep,j:j+yStep,:],i,j
        

    def predict(self, threshold):
        '''
        wsi prediction with trained model image is split 
        into patches if bigger than self.step threshold.
        '''
        _,x,y,_ = K.int_shape(self.image)
        xStep = self.step if x>self.step else x
        yStep = self.step if y>self.step else y

        wsiProbs=np.zeros((x,y,1))
        img=np.zeros((x,y,3))

        for p,i,j in self.split(xStep,yStep):
            with tf.device('/cpu:0'):
                patchProbs=self.model.predict(p)
                wsiProbs[i:i+xStep,j:j+yStep]=patchProbs[:,:]
        self._prediction=tf.cast((wsiProbs>threshold), tf.float32)
        
        return self._prediction


    def thresholdTuning(self,thresholds):
     
        for t in thresholds:
            _ =self.predict(t)
            dice=self.dice
            iou=self.iou
             
            return dice, iou



#modelpath='/home/verghese/breastcancer_ln_deeplearning/scripts/models/unet_germ_2.5x_adam_weightedBinaryCrossEntropy_FRC_data_256_32_18:10.h5'
#modelpath='/home/verghese/breastcancer_ln_deeplearning/scripts/models/multiscale_germ_10x_adam_weightedBinaryCrossEntropy_FRC_data_1024_32_18:36.h5'

modelpath='/home/verghese/breastcancer_ln_deeplearning/scripts/models/multiatten_germ_10x_adam_weightedBinaryCrossEntropy_FRC_data_1024_32_07:49.h5'



imagepath='images/10x/*'
maskpath='masks/10x/*'

images = glob.glob(imagepath)
masks = glob.glob(maskpath)
masks = [m for m in masks if '100188_01_R' not in m]
images = [i for i in images if '100188_01_R' not in i]
model= load_model(modelpath)

print(len(images))
for i in images:
    name=os.path.basename(i)[:-4]
    print(name)
    image=cv2.imread(i)
    image=image/255.0


    mPath=[m for m in masks if name in m][0]
    mask=cv2.imread(mPath)
    mask=mask[:,:,0:1]    


    image=tf.cast(tf.expand_dims(image, axis=0),tf.float32)
    mask=tf.cast(tf.expand_dims(mask, axis=0),tf.float32)
    wsi=WSIPrediction(model,image,mask,2048)
    
    pred = wsi.predict(0.95)
    pred=pred.numpy()
    cv2.imwrite(name+'_pred.png',pred[:,:,0]*255)
    
   

