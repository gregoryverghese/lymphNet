#!/usr/bin/env python3

'''
test.py: script loads trained models and applies them to
unlabelled WSIs. WSIs are split into smaller patches and
then stitched back together
'''

import os
import glob
import json
import argparse

import numpy as np
import openslide
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from prettytable import PrettyTable
import operator
from utilities.augmentation import Normalize
from utilities.utils import resizeImage


def getPatches(slide, w, h, size, mag, magFactor):
    
    for y in range(0,h,size*magFactor):
        for x in range(0,w,size*magFactor):
            patch = slide.read_region((x,y), mag,(size, size))
            patch = patch.convert('RGB')
            yield np.array(patch), x, y

def predict(model,p,xsize,ysize, threshold):
    probs=model.predict(p)
    pred=(probs > threshold).astype(np.int32)
    pred=(pred[0,:,:,:]).astype(np.uint8)
    pred=cv2.resize(pred, (xsize,ysize), interpolation=cv2.INTER_AREA)
    return pred


def applyNormalization(image, normalize=[],channelMeans=None,
                       channelStd=None):
    _=None 
    norm=Normalize(channelMeans, channelStd)
                                               
    for method in normalize:
        image,_=getattr(norm,'get'+method)(image,_)
        return image


def buildSlidePrediction(germModel,sinusModel,slide,mag,
                        magFactor,gThreshold,sThreshold,patchsize,methods,mean,std):

    w,h =slide.dimensions
    hNew=resizeImage(h, patchsize*magFactor, h, operator.gt)
    wNew=resizeImage(w, patchsize*magFactor, w, operator.gt)
    wfactor=wNew/(patchsize*magFactor)
    hfactor=hNew/(patchsize*magFactor)
    wResize=resizeImage(wNew/100, wfactor)
    hResize=resizeImage(hNew/100, hfactor) 
    print('w: {}, wResize: {}, wfinal" {}'.format(w,wResize,int(w/100)))
    print('h: {}, hResize: {}, hfinal" {}'.format(h,hResize,int(h/100)))
    xsize=int(hResize/hfactor)
    ysize=int(wResize/wfactor)
    xfactor=(patchsize*magFactor)/xsize
    yfactor=(patchsize*magFactor)/ysize
    sinus=np.zeros((int(hResize), int(wResize)))
    germinal=np.zeros((int(hResize), int(wResize)))
    temp=np.zeros((int(hResize), int(wResize), 3))

    for p,x,y in getPatches(slide, wNew, hNew, patchsize, mag, magFactor):
        pnew = tf.cast(tf.expand_dims(p,axis=0), tf.float32)
        #pnew = applyNormalization(pnew, normalize=methods,channelMeans=mean,channelStd=std)
        xnew, ynew = int(x/xfactor), int(y/yfactor)
        
        germPred=predict(germModel, pnew,xsize,ysize,gThreshold)
        sinusPred=predict(sinusModel, pnew,xsize,ysize,sThreshold)
        germinal[ynew:ynew+ysize,xnew:xnew+xsize]=germPred[:,:]
        sinus[ynew:ynew+ysize,xnew:xnew+xsize]=sinusPred[:,:]
        p=cv2.resize(p, (xsize,ysize), interpolation=cv2.INTER_AREA)

        temp[ynew:ynew+ysize,xnew:xnew+xsize,0]=p[:,:,0]
        temp[ynew:ynew+ysize,xnew:xnew+xsize,1]=p[:,:,1]
        temp[ynew:ynew+ysize,xnew:xnew+xsize,2]=p[:,:,2]

    hfinal=int(h/100)
    wfinal=int(w/100)

    germinal=germinal[:hfinal,:wfinal]
    sinus=sinus[:hfinal,:wfinal]
    temp=temp[:hfinal,:wfinal]
    
    size=[w,h,hfinal,wfinal]
                        
    return germinal, sinus, temp, size


def test(savePath, wsiPath, germModelPath, sinusModelPath,
         mag, magFactor, sThreshold, gThreshold, downfactor, patchsize):
    

    mean=[0.64,0.39,0.65]
    std=[0.15,0.21,0.18]
    methods=[]
    germModel=load_model(germModelPath)
    sinusModel=load_model(sinusModelPath)
    print(germModel)
    print(sinusModel)
    patients=[p for p in glob.glob(os.path.join(wsiPath, '*'))]
    numPatients = len(patients)
    
    #print(patients)
    totalImages=[]
    for path, subdirs, files in os.walk(wsiPath):
        for name in files:
            if name.endswith('ndpi'):
                totalImages.append(os.path.join(wsiPath, name))

    numImages = len(totalImages)
    avgImgsPatient = int(numImages/numPatients)

    #prettytable to print out key information
    table = PrettyTable(['Patient Number','WSI Number', 'Avg WSI/Patient'])
    table.add_row([numPatients, numImages, avgImgsPatient])
    print(table)
    for p in patients:
        patientId = os.path.basename(p)
        pId=int(patientId.split('.')[0])
        #os.system('mkdir -p ' + os.path.join(savePath,patientId))
        try:
            os.mkdir(os.path.join(savePath,patientId))
        except Exception as e:
            pass
        images = glob.glob(os.path.join(p, '*'))
        numImage = len(images)
        sizeDict={'w':[],'h':[],'hfinal':[],'wfinal':[]}
        for i in range(numImage):
            name = os.path.basename(images[i])[:-5]
            print('name: {}'.format(name))
            try:
                slide = openslide.OpenSlide(images[i])
            except Exception as e:
                print(e)
                print('patient:{}:name{}'.format(patientId,name))
                continue
            #sizeDict={'w':[],'h':[],'hfinal':[],'wfinal':[]}
            germinal, sinus, temp, size = buildSlidePrediction(germModel,sinusModel,slide, 
                                                   mag,magFactor,gThreshold,sThreshold,patchsize,methods,mean,std)
            for i, k in enumerate(sizeDict.keys()):
                sizeDict[k].append(size[i])

            germinal = germinal[:,:,None]*np.ones(3, dtype=int)[None,None,:]
            sinus = sinus[:,:,None]*np.ones(3, dtype=int)[None,None,:]
            
            print('sinus values 1: {}'.format(np.unique(sinus,return_counts=True)))
            sinus[:,:,1]=0
            sinus[:,:,2]=0
            print('sinus values: {}'.format(np.unique(sinus, return_counts=True)))
            germinal[:,:,0]=0
            germinal[:,:,2]=0
            print('germ values', np.unique(germinal, return_counts=True))

            final=germinal+sinus
            final=final.astype(np.uint8)
            temp=temp.astype(np.uint8)
            cv2.imwrite(os.path.join(savePath,patientId,name+'.png'),final*255)
            cv2.imwrite(os.path.join(savePath,patientId,name+'_image.png'),temp)
            cv2.imwrite(os.path.join(savePath,patientId,name+'_imagesinus.png'),sinus*255)
            cv2.imwrite(os.path.join(savePath,patientId,name+'_imagegerm.png'),germinal*255)

        sizeDf=pd.DataFrame(sizeDict)
        sizeDf.to_csv(os.path.join(savePath, 'dimensions.csv'))


if __name__=='__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-cp', '--configpath', required=True, help='config ')

    args=vars(ap.parse_args())
    
    configPath=args['configpath']

    with open(configPath) as jsonFile:
        config=json.load(jsonFile)

    savePath=config['savepath']
    wsiPath=config['wsipath']
    modelPath=config['modelpath']
    germModelName=config['germinalmodel']
    sinusModelName=config['sinusmodel']
    mag=config['mag']
    magFactor=config['magFactor']
    sThreshold=config['sinusThreshold'],
    gThreshold=config['germThreshold']
    downfactor=config['downfactor']
    patchsize=config['patchsize']

    germModelPath=os.path.join(modelPath,germModelName)
    sinusModelPath=os.path.join(modelPath,sinusModelName)

    test(savePath, wsiPath, germModelPath, sinusModelPath,
         mag,magFactor,sThreshold, gThreshold, downfactor, patchsize)
         
         
         
         
         
         

