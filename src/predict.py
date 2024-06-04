"""
predict.py: inference on LNs using trained model
"""


import os
import glob
import argparse
import datetime
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
#import torch
import tensorflow as tf
#import torch.nn.functional as F
#import torch.nn as nn
#from torchvision import transforms as T

from networks.unet_multi import UNet_multi 
from utilities.evaluation import diceCoef
from utilities.augmentation import Augment, Normalize
from stitching import Canvas, stitch

DEBUG = True

def dice_coef(y_true,y_pred,idx=[0,2,3],smooth=1):
        y_true=y_true.type(torch.float16) #float32)
        y_pred=y_pred.type(torch.float16) #float32)
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
        self.tile_dim = tile_dim
        self.step=step
        self.normalize=normalize
        self.channel_means=[float(m) for m in channel_means]
        self.channel_std=[float(s) for s in channel_std]


    def _patching(self, x_dim, y_dim):
        for x in range(0, x_dim-self.step, self.step):
            for y in range(0, y_dim-self.step, self.step):
                x_new = x_dim-self.tile_dim if x+self.tile_dim>x_dim else x
                y_new = y_dim-self.tile_dim if y+self.tile_dim>y_dim else y
                yield x_new, y_new


    def _normalize(self, image, mask):
        norm=Normalize(self.channel_means, self.channel_std)
        data=[(image,mask)]
        #print(self.normalize)
        for method in self.normalize:
            #print(str(method))
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
        margin=int((self.tile_dim-self.step)/2)
        y_dim, x_dim, _ = image.shape

        #using this to call getStandardiseDataset on patches where the image is too big

        c=Canvas(y_dim,x_dim)
        for x, y in self._patching(x_dim,y_dim):
            patch=image[y:y+self.tile_dim,x:x+self.tile_dim,:]
            
            #the np.zeros is a placeholder for the mask - mask doesn't actually get modified here at all
            patch,_ = self._normalize(patch,np.zeros((1,1,1)))

            patch=np.expand_dims(patch,axis=0)
            logits=self.model(patch)
            #print("logits:",logits.shape)
            prediction=tf.cast((logits>self.threshold), tf.float16) #tf.float32)
            #print("preds:",prediction.shape)
            stitch(c, prediction, y, x, y_dim, x_dim, self.tile_dim, self.step, margin)
            #plt.imshow(c.canvas[0,:,:,:])
            #plt.show()
   
        return c.canvas.astype(np.uint8)

    



def test_predictions(model,
                     test_path,
                     save_path,
                     feature,
                     threshold=0.7,
                     tile_dim=1024,
                     step=512,
                     normalize=[],
                     channel_means=[],
                     channel_std=[]
                     ):
    dices=[]
    names=[]

    if DEBUG: print("save path: ",save_path)
    if DEBUG: print("threshold: ",threshold)
    if DEBUG: print("normalize: ",normalize)
    #HR 17/05/23
    #added sorted to make sure we have the right mask to image
    image_paths=sorted(glob.glob(os.path.join(test_path,'images',feature,'*')))
    mask_paths=sorted(glob.glob(os.path.join(test_path,'masks',feature,'*')))
    #if DEBUG: print("mask paths: ",mask_paths)
    #if DEBUG: print("image paths: ",image_paths)
    #if DEBUG: print("means",channel_means)
    #if DEBUG: print("stds",channel_std)
    predict=Predict(model,threshold,tile_dim,step,normalize,channel_means,channel_std)

    for i, i_path in enumerate(image_paths):
        name=os.path.basename(i_path)[:-4]
        print(name)
        names.append(name)
        if DEBUG: print(name)
        m_path=[m for m in mask_paths if name in m][0]
        mask=cv2.imread(m_path)
        print("loaded mask",datetime.datetime.now().strftime('%H:%M'))
        image=cv2.imread(i_path)
        print("\nloaded image",datetime.datetime.now().strftime('%H:%M'))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        print("\nconverted to RGB",datetime.datetime.now().strftime('%H:%M'))
        #image,mask=predict._normalize(image,mask)
        #print("\nnormalised",datetime.datetime.now().strftime('%H:%M'))
        prediction=predict._predict(image)
        print("\ngot predictions",datetime.datetime.now().strftime('%H:%M'))
        mask=np.expand_dims(mask,axis=0)
        print("expanded mask dims",datetime.datetime.now().strftime('%H:%M'))
        #if DEBUG: print("shapes:",prediction.shape,mask.shape)
       
        #print(np.sum(prediction))
        #print(np.sum(mask))
        print("prediction shape: ",prediction.shape)
        print("mask shape: ",mask.shape) 
        dices.append(diceCoef(prediction,mask[:,:,:,0:1]))
        writePredictionsToImage(prediction,save_path,str("pred_"+name))
        writePredictionsToImage(mask,save_path,str("mask_"+name)) 
        print("written ims to file",datetime.datetime.now().strftime('%H:%M'))
        if DEBUG: print(names[i],dices[i])
	#cv2.imwrite(os.path.join(save_path,'predictions',name+'.png'),prediction[0,:,:,:]*255)
    #print(dices)
    #convert list of [1,] tensors to list of floats so easy to view in csv
    da = [dt.numpy() for dt in dices]
    dices_vals = [(list(db))[0] for db in da]
    if DEBUG: print("dice vals:",dices_vals)
    dice_df=pd.DataFrame({'names':names,'dices':dices,'dicevals':dices_vals})
    dice_df.to_csv(os.path.join(save_path,'results-'+str(threshold)+'.csv'))
    #if DEBUG: print(dice_df)
    return dices_vals



## writePredictionsToImage
## helper function to take care of processing and write image in required format
## 21/04/23 Holly Rafique - created function
##
def writePredictionsToImage(img,save_path,name):
    
    img_out = img[0,:,:,:]
    img_out = img_out*255
  
    img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
    #if DEBUG: print("writing to image:",name) 
    #cv2.imwrite(os.path.join(save_path,'predictions',name+".png"),img_out)
  
    cv2.imwrite(os.path.join(save_path,name+".png"),img_out)





if __name__=='__main__':
    ap=argparse.ArgumentParser(description='model inference')
    ap.add_argument('-mp','--model_path',required=True,help='path to trained model')
    ap.add_argument('-tp','--test_path',required=True,help='path to test images and masks')
    ap.add_argument('-sp','--save_path',required=True,help='experiment folder for saving results')
    ap.add_argument('-f','--feature',required=True,help='morphological feature')
    ap.add_argument('-th','--threshold',default=0.75,help='activation threshold')
    ap.add_argument('-td','--tile_dim',default=1024,help='tile dims')
    ap.add_argument('-s','--step',default=512,help='sliding window size')
    ap.add_argument('-n','--normalize',nargs='+',default=["Scale","StandardizeDataset"],help='normalization methods')
    ap.add_argument('-cm','--means',nargs='+',default=[0.675,0.460,0.690],help='channel mean')
    ap.add_argument('-cs','--std',nargs='+', default=[0.180,0.269,0.218],help='channel std')
    args=ap.parse_args()

    #model=UNet_multi(3,2)
    #state_dict=torch.load(args.model_path,map_location='cpu')
    #model.load_state_dict(state_dict)
    model=load_model(args.model_path, compile=False)
    if DEBUG: print(model)
    
    curr_date=str(datetime.date.today())
    curr_time=datetime.datetime.now().strftime('%H:%M')
    cm = [float(x) for x in args.means]
    cs = [float(x) for x in args.std]
    #set up paths for models, training curves and predictions

    save_path = os.path.join(args.save_path,curr_date+"-"+str(args.threshold))
    #save_path = save_path+("_%.2f" % args.threshold)
    if DEBUG: print("save_path:",save_path)
    os.makedirs(save_path,exist_ok=True)
    #os.makedirs(os.path.join(save_path,'predictions'),exist_ok=True)
    test_predictions(model,
                     args.test_path,
                     save_path,
                     args.feature,
                     float(args.threshold),
                     int(args.tile_dim),
                     int(args.step),
                     args.normalize,
                     cm,
                     cs)
    



