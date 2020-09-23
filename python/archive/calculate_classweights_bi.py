import glob
import cv2
import numpy as np
from skimage.transform import resize
from skimage import img_as_bool
from sklearn.utils import class_weight
import pandas as pd
import itertools

masks = glob.glob('/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/masks_g/GERMINAL/*')
print(len(masks))
weights=[]
i=0

for f in masks:
    mask = cv2.imread(f)
    mask = img_as_bool(resize(mask, (224, 224)))
    mask = mask.astype('uint8')
    labels= mask.reshape(-1)
    print(np.unique(labels))
    classWeight = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
    weights.append(classWeight)
    i=i+1
    print(i, flush=True)

a, b = zip(*weights)
finalWeights = [np.mean(a), np.mean(b)]
print(finalWeights, flush=True)

np.savetxt('weights_bi.csv', finalWeights, delimiter=',')
