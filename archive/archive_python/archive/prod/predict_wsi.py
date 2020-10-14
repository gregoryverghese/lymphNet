import os
import glob
import cv2
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
#from evaluation import diceCoef, iouScore
import matplotlib.pyplot as plt



def diceCoef(yTrue, yPred, smooth=1):
    
    yTrue = K.flatten(yTrue)
    yPred = K.flatten(yPred)
    intersection = K.sum(yTrue * yPred)
    return (2. * intersection + smooth) / (K.sum(yTrue) + K.sum(yPred) + smooth)


def iouScore(yTrue, yPred, smooth=1):

    yTrue = K.flatten(yTrue)
    yPred = K.flatten(yPred)
    intersection = K.sum(yTrue*yPred)
    union = K.sum(yTrue) + K.sum(yPred) - intersection
    iou = (intersection + smooth)/(union + smooth)

    return iou



def mergePatches(name, boundaries, modelName, magFactor, imgSize, outPath):

    path = os.path.join(outPath, modelName, name)
    m = ((boundaries[0][1] - boundaries[0][0])//(imgSize*magFactor)*imgSize*magFactor)

    for h in range(boundaries[1][0], boundaries[1][1], imgSize*magFactor):
        for w in range(boundaries[0][0], boundaries[0][1], imgSize*magFactor):


            predPath = os.path.join(path, 'patches', name +'_'+str(w)+'_'+str(h)+'_pred.png')
            patch = cv2.imread(predPath)
            patchNew = cv2.resize(patch, (500,500))

            if w == boundaries[0][0]:
                image = patchNew
            else:
                image = np.hstack((image, patchNew))

        if (w, h) == (boundaries[0][0]+m, boundaries[1][0]):
            final = image
        else:
            final = np.vstack((final, image))

    cv2.imwrite(os.path.join(path, name +'_pred.png'), final)
    cv2.imwrite(os.path.join('output/whole', name +'_pred.png'), final)

    del final
    del image


def readTF(serialized):

    dataMap = {'image': tf.io.FixedLenFeature((), tf.string),
               'mask': tf.io.FixedLenFeature((), tf.string),
               'label': tf.io.FixedLenFeature((), tf.string)}

    example = tf.io.parse_single_example(serialized, dataMap)
    image = tf.image.decode_png(example['image'])
    mask = tf.image.decode_png(example['mask'])
    label = example['label']

    image = tf.reshape(image, (2048, 2048, 3))
    mask = tf.reshape(mask, (2048, 2048, 3))
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)

    return image, mask, label


def patchPredict(dataset, model, modelName, outPath, img):
    diceLst=[]
    iouLst=[]
    for data in dataset:
        image = data[0].numpy()
        mask = data[1].numpy()
        label = data[2].numpy().decode('utf-8')

        figPath = os.path.join(outPath, modelName, img, 'figures')
        outPatchPath = os.path.join(outPath, modelName, img, 'patches')

        image2 = np.expand_dims(image, axis=0)
        image2 = tf.cast(image2, tf.float32)
        probabilities = model.predict(image2)
        prediction = (probabilities > 0.5).astype('int16')

        mask[mask!=0]=1
        prediction[prediction!=0]=1
        prediction1 = prediction[0,:,:,0]
        mask = mask[:,:,0]
        prediction1 = tf.cast(prediction1, tf.float32)

        dice = diceCoef(mask, prediction1)
        diceLst.append(dice.numpy())
        iou = iouScore(mask, prediction1)
        iouLst.append(iou.numpy())

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].set_title(label)
        axs[0].imshow(image)
        axs[0].axis('off')
        axs[1].imshow(mask*255, cmap='gray')
        axs[1].axis('off')
        axs[2].imshow(prediction1*255, cmap='gray')
        axs[2].axis('off')

        fig.savefig(os.path.join(figPath, label + '_pred.png'))
        cv2.imwrite(os.path.join(outPatchPath, label + '_pred.png'), prediction[0,:,:,:]*255)
        plt.close()

    imgscores = pd.DataFrame({'dice':diceLst, 'iou':iouLst})
    imgscores.to_csv(os.path.join(outPatchPath, '_imgscores.csv'))

    avgDice = np.mean(diceLst)
    avgIOU = np.mean(iouLst)
    summary = pd.DataFrame({'dice':[avgDice], 'iou': [avgIOU]})
    summary.to_csv(os.path.join(outPatchPath, '_summary.csv'))



def getWSIPredictions(model, modelName, path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation', tfrecordDir='tfrecords_wsi', outPath='output/predictions', imgSize=2048):

    try:
        os.mkdir(os.path.join(outPath, modelName))
    except:
        pass

    if 'germ' in modelName:
        feature = 'germinal'
    elif 'sinus' in modelName:
        feature = 'sinus'
    elif 'follicle' in modelName:
        feature = 'follicle'

    if '2.5x' in modelName:
        magLevel = '2.5x'
        magFactor = 16
    elif '_5x' in modelName:
        magLevel = '5x'
        magFactor = 8
    elif '_10x' in modelName:
        magLevel = '10x'
        magFactor = 4

    path = os.path.join(path, magLevel, 'one')

    print('paths', os.path.join(path, tfrecordDir, feature, '*'))
    records = glob.glob(os.path.join(path, 'tfrecords', tfrecordDir, feature, '*'))
    print('records', len(records))

    with open('config/config_boundaries.json') as jsonFile:
        boundaries = dict(json.load(jsonFile))

    boundaries = boundaries[magLevel]

    for r in records:
        imgName = os.path.basename(r)[:-10]
        print('Image: {}'.format(imgName))
        
        try:
            os.mkdir(os.path.join(outPath, modelName, imgName))
        except:
            pass

        try:
            os.mkdir(os.path.join(outPath, modelName, imgName, 'patches'))
            os.mkdir(os.path.join(outPath, modelName, imgName,'figures'))
        except:
            print('folders already exist')

        dataset = tf.data.TFRecordDataset(r)
        dataset = dataset.map(readTF, num_parallel_calls=1)
        
        patchPredict(dataset, model, modelName, outPath, imgName)
        mergePatches(imgName, boundaries[imgName], modelName, magFactor, imgSize, outPath)
    
    yTrue = K.flatten(yTrue)
    yPred = K.flatten(yPred)
    intersection = K.sum(yTrue * yPred)
    return (2. * intersection + smooth) / (K.sum(yTrue) + K.sum(yPred) + smooth)


def iouScore(yTrue, yPred, smooth=1):

    yTrue = K.flatten(yTrue)
    yPred = K.flatten(yPred)
    intersection = K.sum(yTrue*yPred)
    union = K.sum(yTrue) + K.sum(yPred) - intersection
    iou = (intersection + smooth)/(union + smooth)
