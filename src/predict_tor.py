"""
predict.py: inference on LNs using trained model
"""


import os
import glob
import argparse

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from utilities.evaluation import diceCoef


test_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/patches/segmentation/10x/one/testing'
#model_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/output/models/2022-04-28/attention_sinus_2.5x_adam_weightedBinaryCrossEntropy_FRC_data4_256_01:19.h5'
model_path='/home/verghese/lymphnode-keras/lymphnode/src-2/models'
#save_path='/home/verghese/lymphnode-keras'
class Predict():
    def __init__(self,model,threshold,step):
        self.model=model
        self.threshold=threshold
        self.step=step


    def _patching(self, x_dim, y_dim):
        for x in range(0, x_dim, self.step):
            for y in range(0, y_dim, self.step):
                yield x, y


    def _predict(self, image):
        y_dim, x_dim, _ = image.shape
        canvas=np.zeros((1,int(y_dim),int(x_dim),1))
        for x, y in self._patching(x_dim,y_dim):
            patch=image[y:y+self.step,x:x+self.step,:]
            patch=np.expand_dims(patch,axis=0)
            probs=self.model.predict(patch)
            prediction=tf.cast((probs>self.threshold), tf.float32)
            canvas[:,y:y+self.step,x:x+self.step,:]=prediction
        return canvas.astype(np.uint8)


def test_predictions(model,test_path,save_path,feature,threshold=0.5,step=512):
    dices=[]
    names=[]
    image_paths=glob.glob(os.path.join(test_path,'images',feature,'*'))
    mask_paths=glob.glob(os.path.join(test_path,'masks',feature,'*'))
    predict=Predict(model,threshold,step)
    for i, (i_path,m_path) in enumerate(zip(image_paths,mask_paths)):
        name=os.path.basename(i_path)
        names.append(name)
        mask=cv2.imread(m_path)
        image=cv2.imread(i_path) 
        prediction=predict._predict(image)
        mask=np.expand_dims(mask,axis=0)
        dices.append(diceCoef(prediction,mask))
        cv2.imwrite(os.path.join(save_path,'predictions',name+'.png'),prediction*255)
        print(names[i],dices[i])
    dice_df=pd.DataFrame({'names':names,'dices':dices})
    dice_df.to_csv(os.path.join(save_path,'results.csv'))
    return dices

  
if __name__=='__main__':
    ap=argparse.ArgumentParser(description='model inference')
    ap.add_argument('-mp','--model_path',required=True,help='path to trained model')
    ap.add_argument('-tp','--test_path',required=True,help='path to test images and masks')
    ap.add_argument('-sp','--save_path',required=True,help='experiment folder for saving results')
    ap.add_argument('-th','--threshold',default=0.5,help='activation threshold')
    ap.add_argument('-s','--step',default=512,help='sliding window size')
    args=ap.parse_args()

    model=load_model(args.model_path)
    test_predictions(model,
                     args.test_path,
                     args.save_path,
                     args.threshold,
                     args.step)
    



