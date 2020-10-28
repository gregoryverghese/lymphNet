#!/usr/bin/env python3

'''
test.py
'''

import os
import glob
import json
import argparse

import numpy as np
import openslide
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from prettytable import PrettyTable
import operator

from utilities.utils import resizeImage


def getPatches(slide, w, h, size, mag, magFactor):
    
    for y in range(0,h,size*magFactor):
        for x in range(0,w,size*magFactor):
            patch = slide.read_region((x,y), mag,(size, size))
            patch = patch.convert('RGB')
            yield np.array(patch), x, y


def predict(model,p,xsize,ysize):

    probs=model.predict(p)
    pred=(probs > threshold).astype(np.int32)
    pred=(pred[0,:,:,:]).astype(np.uint8)
    pred=cv2.resize(pred, (xsize,ysize), interpolation=cv2.INTER_AREA)
    return pred


def buildSlidePrediction(germModel,sinusModel,slide,mag,
                        magFactor,threshold,patchsize):

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
    print('hi')
    temp=np.zeros((int(hResize), int(wResize), 3))
    print('hello')
    for p,x,y in getPatches(slide, wNew, hNew, patchsize, mag, magFactor):
        pnew = tf.cast(tf.expand_dims(p,axis=0), tf.float32)
        xnew, ynew = int(x/xfactor), int(y/yfactor)
        germPred=predict(germModel, pnew,xsize,ysize)
        sinusPred=predict(sinusModel, pnew,xsize,ysize)
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
                        
    return germinal, sinus, temp


def test(savePath, wsiPath, germModelPath, sinusModelPath,
         mag, magFactor, threshold, downfactor, patchsize):
 
    germModel=load_model(germModelPath)
    sinusModel=load_model(sinusModelPath)

    patients=[p for p in glob.glob(os.path.join(wsiPath, '*'))]
    numPatients = len(patients)
         
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
        print(patientId) 
        os.system('mkdir -p ' + os.path.join(savePath,patientId))
        images = glob.glob(os.path.join(p, '*'))
        numImage = len(images)

        for i in range(numImage):
            name = os.path.basename(images[i])[:-5]
            print('name: {}'.format(name))
            try:
                slide = openslide.OpenSlide(images[i])
            except Exception as e:
                print(e)
                print('patient:{}:name{}'.format(patientId,name))
                continue
            
            germinal, sinus, temp = buildSlidePrediction(germModel,sinusModel,slide, 
                                                   mag,magFactor,threshold,patchsize)
            
            germinal = germinal[:,:,None]*np.ones(3, dtype=int)[None,None,:]
            sinus = sinus[:,:,None]*np.ones(3, dtype=int)[None,None,:]
            
            sinus[:,:,1]=0
            sinus[:,:,2]=0
            germinal[:,:,0]=0
            germinal[:,:,2]=0

            final=germinal+sinus
            final=final.astype(np.uint8)
            temp=temp.astype(np.uint8)
            cv2.imwrite(os.path.join(savePath,patientId,name+'.png'),final*255)
            cv2.imwrite(os.path.join(savePath,patientId,name+'_image.png'),temp)


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
    threshold=config['threshold']
    downfactor=config['downfactor']
    patchsize=config['patchsize']

    germModelPath=os.path.join(modelPath,germModelName)
    sinusModelPath=os.path.join(modelPath,germModelName)

    test(savePath, wsiPath, germModelPath, sinusModelPath,
         mag,magFactor,threshold, downfactor, patchsize)
         
         
         
         
         
         

