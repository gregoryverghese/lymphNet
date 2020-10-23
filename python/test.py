#!/usr/bin/env python3

'''
test.py
'''

import os
import glob
import argparse

import numpy as np
import openslide
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from prettytable import PrettyTable
import operator

from utilities.utilities import resizeImage


def getPatches(slide, h, w, size, mag):
    
    for y in range(0,h,size):
        for x in range(0,w,size):
            patch = slide.read_region((x,y), mag,(size, size))
            patch = patch.convert('RGB')
            yield np.array(patch), x, y


def predict(model, p, h, w):

    probs=model.predict(p)
    pred=(probs > threshold).astype(np.int32)
    pred=(pred[0,:,:,:]).astype(np.uint8)
    pred=cv2.resize(pred, (xsize,ysize), interpolation=cv2.INTER_AREA)
        
    xnew, ynew = int(x/xfactor), int(y/yfactor)
    print(xnew, ynew, pred.shape)

    new[ynew:ynew+ysize,xnew:xnew+xsize,0]=pred[:,:]
            
    hfinal=int(h/10)
    wfinal=int(w/10)
    new=new[:hfinal,:wfinal]

    return new


def buildSlidePrediction(germModel,sinusModel,slide,mag,threshold):
    
    w,h =slide.dimensions
    hNew=resizeImage(h, patchsize, h, operator.gt)
    wNew=resizeImage(w, patchsize, w, operator.gt)
    wfactor=wNew/patchsize
    hfactor=hNew/patchsize
    wResize=resizeImage(wNew/10, wfactor)
    hResize=resizeImage(hNew/10, hfactor)
    xsize=int(hResize/hfactor)
    ysize=int(wResize/wfactor)
    xfactor=patchsize/xsize
    yfactor=patchsize/ysize
    germinal = np.zeros((int(hResize), int(wResize), 3))
    sinus = np.zeros((int(hResize), int(wResize), 3))

    for p,x,y in getPatches(slide, wNew, hNew, patchsize, mag):
        p = tf.cast(tf.expand_dims(p,axis=0), tf.float32)
        
        germinal=predict(germModel,p,w,h)
        sinus=predict(sinusModel,p,w,h)
                        
    return germinal, sinus


def test(savePath, wsiPath, germModelPath, sinusModelPath,
         mag, threshold, downfactor, patchsize):
 
    germModel=load_model(germModelPath)
    sinusModelPath=load_model(sinusModelPath)

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
            slide = openslide.OpenSlide(images[i])
            
            germinal, sinus = buildSlidePrediction(slide, mag, threshold)
            cv2.imwrite(os.path.join(savePath,patientId,name+'.png'),temp)


if __name__=='__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-cp', '--configpath', required=True, help='config ')

    args=vars(ap.parse_args())
    
    configPath=args['configPath']

    with open(configPath) as jsonFile:
        config=json.load(jsonFile)

    savePath=config['savepath']
    wsiPath=config['wsipath']
    modelPath=config['modelpath']
    germModelName=config['germinalmodel']
    sinusModelName=config['sinusmodel']
    mag=config['mag']
    threshold=config['threshold']
    downfactor=config['downfactor']
    patchsize=config=['patchsize']

    germModelPath=os.path.join(germModelPath,germModelName)
    sinuModelPath=os.path.join(germModelPath,germModelName)

    test(savePath, wsiPath, germModelPath, sinusModelPath,
         mag, threshold, downfactor, patchsize)
         
         
         
         
         
         

