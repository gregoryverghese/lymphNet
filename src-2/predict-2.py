import os
import glob
import argparse

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from utilities.evaluation import diceCoef


image_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/patches/segmentation/10x/one/testing/images/germinal'
model_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/output/models/2022-04-28/attention_sinus_2.5x_adam_weightedBinaryCrossEntropy_FRC_data4_256_01:19.h5'


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
        canvas=np.zeros((int(y_dim),int(x_dim),3))
        print(canvas.shape)
        for x, y in self._patching(x_dim,y_dim):
            patch=image[x:x+self.step,y:y+self.step,:]
            #patch=np.expand_dims(patch,axis=0)
            #probs=self.model.predict(patch)
            #prediction=tf.cast((probs>self.threshold), tf.float32)
            #canvas[x:x+self.step,y:y+self.step]=prediction[0,:,:,0] 
            canvas[x:x+self.step,y:y+self.step,:]=patch

        return canvas.astype(np.uint8)


def test_predictions(model,image_path,threshold,step):
    
    image_paths=glob.glob(os.path.join(image_path,'*'))
    predict=Predict(model,threshold,step)
    for i, path in enumerate(image_paths):
        print(os.path.basename(path))
        image=cv2.imread(path) 
        prediction=predict._predict(image)
        cv2.imwrite(os.path.join('/home/verghese/lymphnode-keras',str(i))+'.png',prediction)
    print('final prediction', prediction.shape)

model=load_model(model_path)
test_predictions(model,image_path,0.5,1024)


"""  
if __name__=='__main__':
    ap=argparse.ArgumentParser(description='model inference')
    ap.add_argument('-mp','--model_path',required=True,help='path to trained model')
    ap.add_argument('-ip','--image_path',required=True,help='path to test images')
    ap.add_argument('-th','--threshold',default=0.5,help='activation threshold')
    ap.add_argument('-s','--step',default=512,help='sliding window size')
    args=ap.parse_args()

    model=load_model(args.model_path)
    test_predictions(model,args.image_path,args.threshold,args.step)
"""
    



