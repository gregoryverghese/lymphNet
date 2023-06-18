"""
predict.py: inference on LNs using trained model
"""


import os
import glob
import argparse

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
#import torch
import tensorflow as tf
#import torch.nn.functional as F
#import torch.nn as nn
#from torchvision import transforms as T

#from networks.unet_multi import UNet_multi 
from stitching import Canvas, stitch
from utilities.evaluation import diceCoef
from utilities.augmentation import Augment, Normalize

#test_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/patches/segmentation/10x/one/testing'
#model_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/output/models/2022-04-28/attention_sinus_2.5x_adam_weightedBinaryCrossEntropy_FRC_data4_256_01:19.h5'
#save_path='/home/verghese/lymphnode-keras'


def dice_coef(y_true,y_pred,idx=[0,2,3],smooth=1):
        y_true=y_true.type(torch.float32)
        y_pred=y_pred.type(torch.float32)
        intersection=torch.sum(y_true*y_pred,dim=idx)
        union=torch.sum(y_true,dim=idx)+torch.sum(y_pred,dim=idx)
        dice=torch.mean((2*intersection+smooth)/(union+smooth),dim=0)
        return dice


class Predict():
    def __init__(self,
                 model,
                 threshold,
                 tile_dim,
                 step, 
                 normalize=[], 
                 channel_means=[],
                 channel_std=[]):

        self.model=model
        self.threshold=threshold
        self.tile_dim=tile_dim
        self.step=step
        self.normalize=normalize
        self.channel_means=[float(m) for m in channel_means]
        self.channel_std=[float(s) for s in channel_std]


    #def _patching(self, x_dim, y_dim):
        #for x in range(0, x_dim, self.step):
            #for y in range(0, y_dim, self.step):
                #yield x, y
    def _patching(self, x_dim, y_dim, tile_dim):
        for x in range(0, x_dim, self.step):
            for y in range(0, y_dim, self.step):
                x_new = x_dim-tile_dim if x+tile_dim>x_dim else x
                y_new = y_dim-tile_dim if y+tile_dim>y_dim else y
                yield x_new,y_new


    def _normalize(self, image, mask):
   
        norm=Normalize(self.channel_means, self.channel_std)
        data=[(image,mask)]
        for method in self.normalize:
            f=lambda x: getattr(norm,'get'+method)(x[0],x[1])
            data=list(map(f, data))
        image,mask=data[0][0],data[0][1]

        return image,mask

    
    def get_transform(self,img):
        #img = Image.fromarray(img)
        #img=img.convert('RGB')
        tt = T.ToTensor()
        n = T.Normalize((0.7486, 0.5743, 0.7222),(0.0126, 0.0712, 0.0168))
        img = tt(img)
        img = n(img)
        return img

    def _predict(self, image):
        #tt = T.ToTensor() 
        #canvas=np.zeros((1,int(y_dim),int(x_dim),1))
        margin=int((args.tile_dim-args.step)/2)
        c=Canvas(x_dim,y_dim)
        #y_dim, x_dim, _ = image.shape
        #for x, y in self._patching(x_dim,y_dim):
        for x, y in self._patching(x_dim,y_dim,self.tile_dim):
            #patch=image[y:y+self.step,x:x+self.step,:]
            patch=image[y:y+self.tile_dim,x:x+self.tile_dim,:]
            patch=np.expand_dims(patch,axis=0)
            #patch=self.get_transform(patch)
            #patch=torch.unsqueeze(patch,0)
            logits=self.model(patch)
            prediction=tf.cast((logits>self.threshold), tf.float32)
            #probs=F.sigmoid(logits)
            #prediction = torch.ge(probs[:,1:2,:,:],self.threshold).float()
            #prediction=torch.permute(prediction,(0,2,3,1))
            #canvas[:,y:y+self.step,x:x+self.step,:]=prediction
            stitch(
                    c,
                    prediction, 
                    y,
                    x,
                    y_dim,
                    x_dim,
                    self.tile_dim,
                    self.step, 
                    margin
                    )
        return canvas.astype(np.uint8)


def test_predictions(model,
                     test_path,
                     save_path,
                     feature,
                     threshold=0.7,
                     tile_dim,
                     step=1024,
                     normalize=[],
                     channel_means=None,
                     channel_std=None
                     ):
    dices=[]
    names=[]
    print(save_path)
    image_paths=glob.glob(os.path.join(test_path,'images','*'))
    mask_paths=glob.glob(os.path.join(test_path,'masks',feature,'*'))
    print(mask_paths)
    predict=Predict(model,threshold,tile_dim,step,normalize,channel_means,channel_std)
    for i, i_path in enumerate(image_paths):
        name=os.path.basename(i_path)[:-9]
        names.append(name)
        print(name)
        m_path=[m for m in mask_paths if name in m][0]
        mask=cv2.imread(m_path)
        image=cv2.imread(i_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image,mask=predict._normalize(image,mask)
        prediction=predict._predict(image)
        mask=np.expand_dims(mask,axis=0)
        print(prediction.shape,mask.shape)
        dices.append(diceCoef(prediction,mask[:,:,:,0:1]))
        cv2.imwrite(os.path.join(save_path,'predictions',name+'.png'),prediction[0,:,:,:]*255)
    print(dices)
    dice_df=pd.DataFrame({'names':names,'dices':dices})
    dice_df.to_csv(os.path.join(save_path,'results.csv'))
    #print(dice_df)
    return dices

  
if __name__=='__main__':
    ap=argparse.ArgumentParser(description='model inference')
    ap.add_argument('-mp','--model_path',required=True,help='path to trained model')
    ap.add_argument('-tp','--test_path',required=True,help='path to test images and masks')
    ap.add_argument('-sp','--save_path',required=True,help='experiment folder for saving results')
    ap.add_argument('-f','--feature',required=True,help='morphological feature')
    ap.add_argument('-th','--threshold',default=0.5,help='activation threshold')
    ap.add_argument('-td','--tile_dim',default=1600,help='tile dims')
    ap.add_argument('-s','--step',default=600,help='sliding window size')
    ap.add_argument('-n','--normalize',nargs='+',default=["Scale"],help='normalization methods')
    ap.add_argument('-cm','--means',nargs='+',default=[0.633,0.383,0.659],help='channel mean')
    ap.add_argument('-cs','--std',nargs='+', default=[0.143,0.197,0.19],help='channel std')
    args=ap.parse_args()

    #model=UNet_multi(3,2)
    #state_dict=torch.load(args.model_path,map_location='cpu')
    #model.load_state_dict(state_dict)
    model=load_model(args.model_path)
    print(model)
    test_predictions(model,
                     args.test_path,
                     args.save_path,
                     args.feature,
                     float(args.threshold),
                     int(args.tile_dim),
                     int(args.step),
                     args.normalize,
                     args.means,
                     args.std)
    



