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
from tensorflow.keras.models import load_model
from prettytable import PrettyTable

from utilities.utilities import resizeImage


def getPatches(slide, h, w, size, magLevel):
     for y in range(0,h,size):
         for x in range(0,w,size):
             patch = patch.read_region((y,x), magLevel,size)
             patch.convert('RGB')
             yield np.array(patch)


def test(wsiPath,modelPath):
    
    mag = 0
    size = 2048
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

            print('h',h,'hnew',hNew,'w',w, 'wnew', wNew)

            patches = [[patch.read_region((x,y), mag, size) for x in wNew] for y in hNew]
            
            #patches [[p] for p in getPatches(slide,hNew, wNew, 2048,0)]
            #patches = [tf.convert_to_tensor(np.array(p.convert('RGB'))) 
            #                                                                for p in patches]
            #print(len(patches))
            #probabilities = [model.predict(p) for p in patches]
            #prediction=list(map(lambda x: x>threshold, probabilities))
            prediction=np.vstack([np.hstack(i) for i in probabilities])

            cv2.imwrite(os.path.join(savePath,patientId,name+'.png'),prediction)

if __name__=='__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-wp', '--wsipath',required=True, help='path to records')
    ap.add_argument('-mp','--modelpath',required=True, help='path to model')

    args=vars(ap.parse_args())
    
    wsiPath=args['wsipath']
    modelPath=args['modelpath']

    test(wsiPath, modelPath)
