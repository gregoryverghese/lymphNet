import os
import glob

import numpy as np

import cv2
import matplotlib.pyplot as plt
import openslide
import pandas as pd

import measure as me

wsiPath='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/testing/wsi'
maskPath='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/testing/output'

totalImages=[]
totalMasks=[]

for path, subdirs, files in os.walk(wsiPath):
    for name in files:
        if name.endswith('ndpi'):
            totalImages.append(os.path.join(wsiPath,name))

for path, subdirs, files in os.walk(maskPath):
    for name in files:
        if name.endswith('png'):
            totalMasks.append(os.path.join(maskPath,name))



print(len(totalImages))
print(len(totalMasks))

