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
from utilities.augmentation import Augment, Normalize

#test_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/patches/segmentation/10x/one/testing'
#model_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/output/models/2022-04-28/attention_sinus_2.5x_adam_weightedBinaryCrossEntropy_FRC_data4_256_01:19.h5'
#save_path='/home/verghese/lymphnode-keras'
class Predict():
    def __init__(self,model,
                 threshold,
                 step, 
                 normalize=[], 
                 channel_means=[],
                 channel_std=[]):

        self.model=model
        self.threshold=threshold
        self.step=step
        self.normalize=normalize
        self.channel_means=[float(m) for m in channel_means]
        self.channel_std=[float(s) for s in channel_std]


    def _patching(self, x_dim, y_dim):
        for x in range(0, x_dim, self.step):
            for y in range(0, y_dim, self.step):
                yield x, y


    def _normalize(self, image, mask):
   
        norm=Normalize(self.channel_means, self.channel_std)
        data=[(image,mask)]
        for method in self.normalize:
            f=lambda x: getattr(norm,'get'+method)(x[0],x[1])
            data=list(map(f, data))
        image,mask=data[0][0],data[0][1]

        return image,mask


    def _predict(self, image):
        
        print("\n\n**** HOLLY ****   in _predict")
        print("\n**** HOLLY **** shape :")
        print(image.shape)
        y_dim, x_dim, _ = image.shape
        canvas=np.zeros((1,int(y_dim),int(x_dim),1))
        for x, y in self._patching(x_dim,y_dim):
            patch=image[y:y+self.step,x:x+self.step,:]
            patch=np.expand_dims(patch,axis=0)
            probs=self.model.predict(patch)
            prediction=tf.cast((probs>self.threshold), tf.float32)
            print("\n**** HOLLY **** x:"+str(x)+" y:"+str(y))
            print("\n**** HOLLY **** probs: ")
            print(probs)
            print("\n**** HOLLY **** prediction: ")
            print(prediction)
            canvas[:,y:y+self.step,x:x+self.step,:]=prediction
            print("\n**** HOLLY **** canvas: ")
            print(canvas)
        print("\n**** HOLLY **** final canvas: ")
        print(canvas)
        print("\n**** HOLLY **** ENDING _predict")
        return canvas.astype(np.uint8)


def test_predictions(model,
                     test_path,
                     save_path,
                     feature,
                     threshold=0.5,
                     step=512,
                     normalize=[],
                     channel_means=None,
                     channel_std=None
                     ):
    dices=[]
    names=[]
    image_paths=glob.glob(os.path.join(test_path,'images',feature,'*'))
    mask_paths=glob.glob(os.path.join(test_path,'masks',feature,'*'))
    ## HOLLY DEBUG MODE
    #image_paths = image_paths[:10]
    #mask_paths = mask_paths[:10] 
    print("\n\n**** HOLLY ****   created predict object")
    predict=Predict(model,threshold,step,normalize,channel_means,channel_std)

    for i, (i_path,m_path) in enumerate(zip(image_paths,mask_paths)):
        print("\n\n**** HOLLY ****   PREDICT LOOP: "+str(i))
        print(i_path)
        print(m_path)

        name=os.path.basename(i_path)
        names.append(name)
        mask=cv2.imread(m_path)
        #do we also need to convert the mask to RGB?        

        print("\n\n**** HOLLY ****   calling _predict")
        image=cv2.imread(i_path)
        image=cv2.cvtColor(image,#cv2.COLOR_BGR2RGB)
        image,mask=predict._normalize(image,mask)
        prediction=predict._predict(image)
        mask=np.expand_dims(mask,axis=0)
        
        dices.append(diceCoef(prediction,mask))

        writePredictionsToImage(prediction,save_path,name)        
        #DEBUG writing mask images out 
        print(str("mask"+name))
        writePredictionsToImage(mask,save_path,str("mask"+name))
        print(names[i],dices[i])
    
    print("\n\n**** HOLLY ****   names")
    print(names)
    print("\n\n**** HOLLY ****   dices:")
    print(dices)
        #cv2.imwrite(os.path.join(save_path,'predictions',name+'.png'),prediction[0,:,:,:]*255)
    dice_df=pd.DataFrame({'names':names,'dices':dices})
    dice_df.to_csv(os.path.join(save_path,'results.csv'))
    print(dice_df)
    return dices

def writePredictionsToImage(img,save_path,name):
    print("\n**** HOLLY **** calling writePredictionsToImage: "+str(os.path.join(save_path,'predictions',name)))
    #need to remove .png because this is already included in os.path.basename
    print("\n**** HOLLY **** prediction: ")
    print(img*255)
    print("\n*** HOLLY *** shape of prediction img: ")
    print(img.shape)
    img_out = img[0,:,:,:]
    img_out = img_out*255
    print("\n*** HOLLY *** img_out before colour space conversion: ")
    print(img_out)
    print("\n*** HOLLY *** shape of img_out: ")
    print(img_out.shape)

    img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

    print("\n*** HOLLY *** final img_out: ")

    print(img_out)
    cv2.imwrite(os.path.join(save_path,'predictions',name),img_out)


  
if __name__=='__main__':
    print("in predict.py")
    ap=argparse.ArgumentParser(description='model inference')
    ap.add_argument('-mp','--model_path',required=True,help='path to trained model')
    ap.add_argument('-tp','--test_path',required=True,help='path to test images and masks')
    #ap.add_argument('-op','--save_path',required=True,help='experiment folder for saving results')
    #ap.add_argument('-f', '--feature', default="germinal",help='feature')
    #ap.add_argument('-th','--threshold',default=0.95,help='activation threshold')
    #ap.add_argument('-s','--step',default=1024,help='sliding window size')
    
    ap.add_argument('-sp','--save_path',required=True,help='experiment folder for saving results')
    ap.add_argument('-f','--feature',required=True,help='morphological feature')
    ap.add_argument('-th','--threshold',default=0.5,help='activation threshold')
    ap.add_argument('-s','--step',default=512,help='sliding window size')
    ap.add_argument('-n','--normalize',nargs='+',default=["Scale"],help='normalization methods')
    ap.add_argument('-cm','--means',nargs='+',default=[0.633,0.383,0.659],help='channel mean')
    ap.add_argument('-cs','--std',nargs='+', default=[0.143,0.197,0.19],help='channel std')
    args=ap.parse_args()
    print("parsed args")
    print("mp: "+args.model_path)
    print("tp: "+args.test_path)
    print("op: "+args.save_path)
    print("f: "+args.feature)
    print("th: "+str(args.threshold))
    print("s: "+str(args.step))

    model=load_model(args.model_path)
    print("loaded model")
    test_predictions(model,
                     args.test_path,
                     args.save_path,
                     args.feature,
                     float(args.threshold),
                     int(args.step),
                     args.normalize,
                     args.means,
                     args.std)
    



