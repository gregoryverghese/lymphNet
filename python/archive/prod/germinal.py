import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import glob
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

'''
def diceCoef(yTrue, yPred, smooth=1):

    intersection = K.sum(yTrue * yPred, axis=[1, 2])
    union = K.sum(yTrue, axis=[1, 2]) + K.sum(yPred, axis=[1, 2])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice
'''

'''
def diceCoef(yTrue, yPred, smooth=1):

    yTrue = K.flatten(yTrue)
    yPred = K.flatten(yPred)
    intersection = K.sum(yTrue * yPred)
    return (2. * intersection + smooth) / (K.sum(yTrue) + K.sum(yPred) + smooth)
'''


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


def calculateWeightedBinaryCrossEntropy(yTrue, yPred, posWeight):
    yPred = tf.clip_by_value(yPred, K.epsilon(), (1-K.epsilon()))
    logits = lossWithLogits(yPred)
    return tf.nn.weighted_cross_entropy_with_logits(logits=logits, labels=yTrue, pos_weight=posWeight)


def calculateWeightedCategoricalCrossEntropy(yTrue, yPred):
    yTrueMask = K.clip(yTrue, 0.0, 1.0)
    cce = categorical_crossentropy(yPred, yTrueMask)
    yTrueWeightsMaxed = K.max(yTrue, axis=-1)
    wcce = cce * yTrueWeightsMaxed
    return K.sum(wcce)


def calculateFocalLoss(yTrue, yPred, alpha, gamma):
    yPred = tf.clip_by_value(yPred, K.epsilon(), 1 - K.epsilon())
    logits = lossWithLogits(yPred)
    weightA = alpha * tf.math.pow((1 - yPred), gamma) * yTrue
    weightB = tf.math.pow((1 - alpha) * yPred, gamma) * (1 - yTrue)
    return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits))*(weightA + weightB) + logits * weightB


def weightedCategoricalCrossEntropy(yTrue, yPred):
    def loss(yTrue, yPred):
        return calculateWeightedCategoricalCrossEntropy(yTrue, yPred)
    return loss


def diceLoss(yTrue, yPred):
    return 1.0 - diceCoef(yTrue, yPred)


def lossWithLogits(yPred):
    return tf.math.log(yPred/(1-yPred))


def focalLoss(alpha=0.25, gamma=2):
    def loss(yTrue, yPred):
        return calculateFocalLoss(yTrue, yPred, alpha, gamma)
    return loss


def weightedBinaryCrossEntropy(posWeight=1):
    def loss(yTrue, yPred):
        return calculateWeightedBinaryCrossEntropy(yTrue, yPred, posWeight)
    return loss


def diceBCELoss(yTrue, yPred, posWeight):
    def loss(yTrue, yPred):
        wbce = calculateWeightedBinaryCrossEntropy(yTrue, yPred, posWeight)
        normWBCE=((1/float(wbce))*wbce)
        dice = diceLoss(yTrue, yPred)
        return normWBCE + dice
    return loss


def wrapBytes(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(imageFiles, imgName, magLevel, feature):

    serialData = []

    for i in imageFiles:
        m = os.path.join(maskPath, str(magLevel), feature, imgName, os.path.basename(i)[:-4]+'_masks.png')
        image = tf.keras.preprocessing.image.load_img(i)
        image = tf.keras.preprocessing.image.img_to_array(image, dtype=np.uint8)
        image = tf.image.encode_png(image)

        mask = tf.keras.preprocessing.image.load_img(m)
        mask = tf.keras.preprocessing.image.img_to_array(mask, dtype=np.uint8)
        mask = tf.image.encode_png(mask)

        label = os.path.basename(i)[:-4].encode('UTF-8')

        data = {'image': wrapBytes(image), 'mask': wrapBytes(mask), 'label': wrapBytes(label)}
        features = tf.train.Features(feature=data)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()

        serialData.append(serialized)

    return serialData


def mergePatches(ndpi, boundaries, modelName, magFactor, imgSize, outPath):

    name = os.path.basename(ndpi)
    path = os.path.join(outPath, modelName, name)
    m = ((boundaries[0][1] - boundaries[0][0])//(imgSize*magFactor)*imgSize*magFactor)

    for h in range(boundaries[1][0], boundaries[1][1], imgSize*magFactor):
        for w in range(boundaries[0][0], boundaries[0][1], imgSize*magFactor):


            predPath = os.path.join(path, 'patches', name +'_'+str(w)+'_'+str(h)+'_pred.png')
            print(predPath)
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


def patchPredict(images, model, feature, modelName, magLevel, magFactor, patchPath, maskPath, outPath):

    allDiceLst=[]
    allIOULst = []
    imageLst = []
    modelLst = []
    zooms = []

    images = [os.path.basename(i) for i in images]

    for imgName in images:
        imageLst = []
        labels = []
        maskLst = []
        diceLst = []
        iouLst = []
        imgPatchPath = os.path.join(patchPath, str(magLevel), imgName)
        patchFiles = glob.glob(os.path.join(imgPatchPath, "*"))
        print('patchFiles', patchFiles)
        try:
            os.mkdir(os.path.join(outPath, modelName, imgName))
        except:
            pass

        try:
            os.mkdir(os.path.join(outPath, modelName, imgName, 'patches'))
            os.mkdir(os.path.join(outPath, modelName, imgName,'figures'))
        except:
            print('folders already exist')

        data = convert(patchFiles, imgName, magLevel, feature)

        for d in data:
            dataMap = {'image': tf.io.FixedLenFeature((), tf.string),
                       'mask': tf.io.FixedLenFeature((), tf.string),
                       'label': tf.io.FixedLenFeature((), tf.string)
                      }

            example = tf.io.parse_single_example(d, dataMap)
            image = tf.image.decode_png(example['image'])
            mask = tf.image.decode_png(example['mask'])
            label = example['label'].numpy().decode('utf-8')
            imageLst.append(image.numpy())
            maskLst.append(mask.numpy())
            labels.append(label)

        imageArr =np.array(imageLst)
        maskArr = np.array(maskLst)

        figPath = os.path.join(outPath, modelName, imgName,'figures')
        outPatchPath = os.path.join(outPath, modelName, imgName, 'patches')

        for i, (image, mask) in enumerate(zip(imageArr, maskArr)):
            
            print('labels', labels[i])
            image = image.astype(np.int32)
            mask = mask.astype(np.int32)
            image2 = np.expand_dims(image, axis=0).astype(np.float16)

            predProbs = model.predict(image2)
            prediction = (predProbs > 0.5).astype('int16')

            mask[mask!=0]=1
            mask = tf.cast(mask[:,:,0], tf.float32)
            prediction = tf.cast(prediction[0,:,:,0], tf.float32)
            dice = diceCoef(mask, prediction)
            diceLst.append(dice.numpy())
            iou = iouScore(mask, prediction)
            iouLst.append(iou.numpy())

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            axs[0].set_title(labels[i])
            axs[0].imshow(image)
            axs[0].axis('off')
            axs[1].imshow(mask*255, cmap='gray')
            axs[1].axis('off')
            axs[2].imshow(prediction*255, cmap='gray')
            axs[2].axis('off')


            prediction = (prediction.numpy()*255).astype(np.uint8)
            fig.savefig(os.path.join(figPath, labels[i] + '_pred.png'))
            cv2.imwrite(os.path.join(outPatchPath, labels[i] + '_pred.png'), prediction)
            plt.close()

        avgDice = np.mean(np.array(diceLst))
        avgIOU = np.mean(np.array(diceLst))

        imgscores = pd.DataFrame({'dice':diceLst, 'iou':iouLst})
        imgscores.to_csv(os.path.join(outPatchPath, '_imgscores.csv'))
        summary = pd.DataFrame({'dice':[avgDice], 'iou': [avgIOU]})
        summary.to_csv(os.path.join(outPatchPath, '_summary.csv'))

        allDiceLst.append(avgDice)
        allIOULst.append(avgIOU)
        imageLst.append(imgName)
        modelLst.append(modelName)
        zooms.append(magLevel)

    return allDiceLst, allIOULst, imageLst, modelLst, zooms


def getWSIPredictions(feature, outPath, modelPath, patchPath, maskPath):

    allModels=[]
    allImages=[]
    allIous=[]
    allDices=[]
    zooms=[]

    zooms = glob.glob(os.path.join(patchPath, '*'))
    zooms = [os.path.basename(z) for z in zooms]
    zooms = ['4']

    for z in zooms:
        print('Zoom: {}'.format(z), flush=True)
        if z == '4':
            magLevel = 4
            magFactor = 16
            imgSize = 2048
            boundaries={'U_100246_10_B_LOW_10_L1':[[43000, 94800], [37000, 54400]], 'U_100278_6_B_LOW_6_L1':[[13600, 47600], [43200, 61200]], 'U_100237_22_X_LOW_11_L1':[[53239, 63554], [24734, 41667]]}
        elif z == '3':

            magLevel = 3
            magFactor = 8
            imgSize = 2048
            boundaries = {'U_100246_10_B_LOW_10_L1':[[43000, 94800], [37000, 54400]],'U_100278_6_B_LOW_6_L1': [[13600, 47600], [43200, 61200]], 'U_100237_22_X_LOW_11_L1': [[53200, 63600], [24600, 41800]]}

        models = glob.glob(os.path.join(modelPath, z, feature, '*'))
        models = [m for m in models if '.h5' in m]
        check = glob.glob('output/test/*')
        check = [os.path.basename(c[:-3]) for c in check]
        print(check)

        for m in models:
            modelName = os.path.basename(m)[:-3]
            if modelName not in check:

                print('Model: {}'.format(modelName), flush=True)

                try:
                    os.mkdir(os.path.join(outPath, modelName))
                except:
                    print('model folder already exists')

                if 'weightedBinaryCrossEntropy' in modelName:
                    lossKey = 'loss'
                    lossFunc = weightedBinaryCrossEntropy(1)
                elif 'diceloss' in modelName:
                    lossKey = 'diceLoss'
                    lossFunc = diceLoss

                dependencies = {'diceCoef':diceCoef, lossKey:lossFunc}
                model = load_model(m, custom_objects=dependencies)
                images = glob.glob(os.path.join(patchPath, str(magLevel), '*'))

                allDiceLst, allIOULst, imageLst, modelLst, zooms = patchPredict(images, model, feature, modelName, magLevel, magFactor, patchPath, maskPath, outPath)

                for img in images:
                    mergePatches(img, boundaries[os.path.basename(img)], modelName, magFactor, imgSize, outPath)

                allModels+= modelLst
                allImages+= imageLst
                allIous+= allIOULst
                allDices+= allDiceLst



    summary = pd.DataFrame({'model':allModels, 'image':allImages, 'dice':allDices, 'iou':allIous})
    summary.to_csv(feature+'.csv')


modelPath = 'output/models2'
patchPath = 'output/patches'
maskPath = 'output/masks'
outPath = 'output/test'
feature = 'follicle'

w = getWSIPredictions(feature, outPath, modelPath, patchPath, maskPath)
