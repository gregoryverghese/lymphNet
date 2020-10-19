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

from utilities.utilities import resizeImage


def getPatches(slide, h, w, size, magLevel):

    for x in range(0,w,size):
        for y in range(0,h,size):
            patch = slide.read_region((x,y), 0,(size, size))
            patch = patch.convert('RGB')
            yield np.array(patch), x, y


def test(wsiPath,modelPath):
    threshold = 0.5
    downfactor = 10
    modelPath = os.path.join(modelPath, 'unet_germ_2.5x_adam_weightedBinaryCrossEntropy_FRC_17:40.h5')
    mag = 0
    size = 4096
    model=load_model(modelPath)
    totalImages = []

    patients=[p for p in glob.glob(os.path.join(wsiPath, '*'))]
    numPatients = len(patients)

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

        try:
           os.mkdir(os.path.join())
        except Exception as e:
           pass

        images = glob.glob(os.path.join(p, '*'))
        numImage = len(images)

        for i in range(numImage):
            name = os.path.basename(images[i])[:-5]
            slide = openslide.OpenSlide(images[i])
            
            h, w = slide.dimensions
            hNew = resizeImage(h)
            wNew = resizeImage(w)
            factor = wNew/size
            wResize=resizeImage(wNew/10, factor)
            hResize=resizeImage(hNew/10, factor)
            wResize2=resizeImage(wResize, factor)
            hResize2=resizeImage(hResize, factor)
            temp = np.zeros((int(wResize), int(hResize), 3))

            for p,x,y in getPatches(slide, wNew, hNew, size, 0):
                p = tf.cast(tf.expand_dims(p,axis=0), tf.float32)

                print(p.shape)
                probabilities = model.predict(p)
                prediction = (probabilities > threshold).astype(np.int32)
                print('prediction shape', prediction.shape)
                print((int(wResize/factor)))
                print(prediction)
                prediction = (prediction[0,:,:,0]).astype(np.uint8)
                print(prediction.shape)
                prediction=cv2.resize(prediction, (int(wResize2/factor), int(hResize2),interpolation=cv2.INTER_AREA)
                temp[y:y+size,x:x+size,0]=prediction[:,:,0]
                temp[y:y+size,x:x+size,1]=prediction[:,:,1]
                temp[y:y+size,x:x+size,1]=prediction[:,:,2]

            cv2.imwrite(os.path.join(savePath,patientId,name+'.png'),prediction)



if __name__=='__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-wp', '--wsipath',required=True, help='path to records')
    ap.add_argument('-mp','--modelpath',required=True, help='path to model')

    args=vars(ap.parse_args())
    
    wsiPath=args['wsipath']
    modelPath=args['modelpath']

    test(wsiPath, modelPath)
         
         
         
         
         
         

