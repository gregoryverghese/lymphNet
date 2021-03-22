import os
import glob

import cv2
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations, product

def diceCoef(yTrue, yPred, axIdx=(1,2,3), smooth=1):

    yPred = yPred.astype(np.float32)
    yTrue = yTrue.astype(np.float32)

    intersection = np.sum(yTrue * yPred, axis=axIdx)
    union = np.sum(yTrue, axis=axIdx) + np.sum(yPred,axis=axIdx)
    dice = np.mean((2. * intersection + smooth)/(union + smooth), axis=0)

    return dice

path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/testing/output/quantify_plots/*'
images=glob.glob(path)
images=[i for i in images if 'binarymask' in i]
df = pd.read_csv('/home/verghese/patients.csv')
print(df.columns)
patients=list(df['patients'])
patients=[p.split('.')[1].strip() for p in patients]
print(patients)
print('Number of images: {}'.format(len(images)))
print(df)

names1=[]
names2=[]
info=[]
dices=[]
outliers1=[]
outliers2=[]
diceoutliers=[]
patientoutliers=[]

for p in patients:
    patient_images=[i for i in images if p in i]
    combs=list(product(patient_images, patient_images))
    for c in combs:
        print(c)

        name1=os.path.basename(c[0])[:-15]
        name2=os.path.basename(c[1])[:-15]

        image1=cv2.imread(c[0])
        image2=cv2.imread(c[1])

        max_x=max(image1.shape[0],image2.shape[0])
        max_y=max(image1.shape[1],image2.shape[1])

        pad_x=max_x-image1.shape[0]
        pad_y=max_y-image1.shape[1]

        test1_new=np.pad(image1,((pad_x,0),(0,pad_y),(0,0)),'constant')

        pad_x=max_x-image2.shape[0]
        pad_y=max_y-image2.shape[1]

        test2_new=np.pad(image2,((pad_x,0),(0,pad_y),(0,0)),'constant')
        test1=np.expand_dims(test1_new,axis=0)
        test2=np.expand_dims(test2_new,axis=0)

        test1[test1==255]=1
        test2[test2==255]=1

        channel1 = np.zeros(test1.shape[:-1]+(1,), dtype=test1.dtype)
        channel2 = np.zeros(test2.shape[:-1]+(1,), dtype=test2.dtype)

        test1 = np.concatenate((test1, channel1), axis=-1)
        test2 = np.concatenate((test2, channel2), axis=-1)

        test1[:,:,:,3][test1[:,:,:,0]==0]=1
        test2[:,:,:,3][test1[:,:,:,0]==0]=1

        weights1=np.sum(test1[0,:,:,:],axis=(0,1))/np.sum(test1)
        weights2=np.sum(test2[0,:,:,:],axis=(0,1))/np.sum(test2)
        weights=(weights1+weights2)*0.5

        lst=[]
        for i in range(4):
            w=weights[i]
            lst.append(diceCoef(test1[:,:,:,i:i+1],test2[:,:,:,i:i+1]))

        dice=np.mean(lst)

        info.append((dice,c))
        names1.append(name1)
        names2.append(name2)
        dices.append(dice)

        if dice>0.7 and dice!=1.0:
            patientoutliers.append(p)
            outliers1.append(name1)
            outliers2.append(name2)
            diceoutliers.append(dice)
            cv2.imwrite('levels/'+name1+'.png',image1)
            cv2.imwrite('levels/'+name2+'.png',image2)

df=pd.DataFrame({'patients':patientoutliers,'outliers1':outliers1,'outliers2':outliers2,'diceoutliers':diceoutliers})
df.to_csv('outliers.csv')
