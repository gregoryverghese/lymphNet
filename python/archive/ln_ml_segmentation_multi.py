import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import itertools
import glob
import shutil
import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from skimage.filters import sobel, prewitt, scharr
from skimage.transform import resize
from skimage import img_as_bool
from image_feature_engineering import FeatureEngineering, GaborFilters


def getImageMask(filePath, maskPath, basePath, imgSize=224):
    try:
        img = cv2.imread(filePath)
        img = cv2.resize(img, (imgSize, imgSize))

        maskFile = os.path.basename(filePath)[:-4] +'_masks.png'
        maskPath = os.path.join(maskPath, maskFile)

        mask = cv2.imread(maskPath)
        mask = mask.astype(np.bool)
        mask = img_as_bool(resize(mask,  (224, 224)))
        mask = mask.astype('uint8')

        labels = np.zeros((224, 224))

        for c in range(3):
            labels[mask[:,:,c]==True]=c

        return img, labels
    except cv2.error as e:
            return None, None
    


def featureExtractor(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fe = FeatureEngineering(img)

    features = GaborFilters(img).getGabors()
    features.append(fe.getCanny())
    features.append(fe.getRoberts())
    features.append(fe.getSobel())
    features.append(fe.getGaussian(3))
    features.append(fe.getGaussian(5))
    features.append(fe.getGaussian(7))
    features.append(fe.getScharr())
    features.append(fe.getPrewitt())
    features.append(fe.getMedian(size=3))

    return features


def getFeatures(imgFiles, maskPath, basePath, imgSize=224):

    i=0
    allLabels = []
    allFeatures = []

    for file in imgFiles:
        print(i, flush=True)
        img, mask = getImageMask(file, maskPath, basePath, imgSize=224)
        if img is None:
            continue
        allFeatures.append(featureExtractor(img))
        allLabels.append(mask.reshape(-1))
        i=i+1

    allFeatures = [np.hstack(f) for f in zip(*allFeatures)]
    allFeatures = np.transpose(np.array(allFeatures))
    allLabels = np.hstack(allLabels)

    return allFeatures, allLabels


def trainClassifier(X, y, n=10, randomState=42):

    model = RandomForestClassifier(n_estimators=n, random_state=randomState)
    model.fit(X, y)

    return model


def getPredictions(model, testFiles, maskPath, basePath, outPath):

    pixelPredictions=[]
    pixelLabels=[]
    accuracy=[]
    precision=[]
    recall=[]
    f1=[]
    roc_auc=[]
    cm=np.zeros((3,3))

    for fi in testFiles:
        print(os.path.join(maskPath, os.path.basename(fi)[:-4]))

        img, mask = getImageMask(fi, maskPath, basePath, imgSize=224)
        if img is None:
            continue
        f = featureExtractor(img)
        f = np.array(f)
        f = np.transpose(f)

        pred = model.predict(f)
        labels = mask.reshape(-1)
        maskPred = pred.reshape((224, 224))
       
        accuracy.append(accuracy_score(labels, pred))
        precision.append(precision_score(labels, pred, average='weighted'))
        f1.append(f1_score(labels, pred, average='weighted'))
        recall.append(recall_score(labels, pred, average='weighted'))
        roc_auc.append(roc_auc_score(labels, pred, average='weighted'))

        maskPred = np.uint8(maskPred)
        contours, hierarchy = cv2.findContours(maskPred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours)==0:
            continue

        #maxContour = max(contours, key=cv2.contourArea)
        imgPred = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)
        imgPred = cv2.resize(imgPred, (1000,1000))

        mask = np.uint8(mask)
        maskContour, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(maskContour)==0:
            continue

        #maxMaskContour = max(maskContour, key=cv2.contourArea)
        imgTrue = cv2.drawContours(img, maskContour, -1, (255, 0, 0), 3)
        imgTrue = cv2.resize(imgPred, (1000,1000))

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)*255
        maskPred = cv2.cvtColor(maskPred, cv2.COLOR_GRAY2BGR)*255
        mask = cv2.resize(mask, (1000,1000))
        maskPred = cv2.resize(maskPred, (1000,1000))
        combined = np.concatenate((imgTrue, mask, maskPred, imgPred), axis=1)

        cv2.imwrite(os.path.join(outPath, os.path.basename(fi)) + '_annot.png', combined)
        #cv2.imwrite(os.path.join('output/output',os.path.basename(file)) + '_mask.png', maskPred)

        #plt.figure(figsize=(10,10))
        #plt.imshow(combined)
        #plt.axis('off')
        #plt.show()

    #pixPredictions = np.hstack(pixelPredictions)
    #pixLabels = np.hstack(pixelLabels)
    evaluationDf = pd.DataFrame({'accuracy': accuracy, 'precision':precision, 'f1':f1, 'recall':recall, 'roc_auc':roc_auc})
    #resultDf = pd.DataFrame({'predictions':pixPredictions, 'labels':pixLabels})

    evaluationDf.to_csv('evaluation_multi.csv')
    #resultDf.to_csv('result.csv')


def getSegmentation(basePath, imagePath, maskPath, outPath, testSize=0.1):

    classPaths = glob.glob(os.path.join(imagePath, '*'))

    #if len(classPaths)<2:
        #gen = dataGenerator(basePath, imagePath, maskPath)
        #gen.imageGenerator()
        #gen.maskGenerator()

    imgFiles  = glob.glob(os.path.join(imagePath, '*'))
    imgTrain, imgTest = train_test_split(imgFiles, test_size=testSize, random_state=20)
    trainFeatures, trainLabels = getFeatures(imgTrain, maskPath, basePath)
    model = trainClassifier(trainFeatures, trainLabels)
    pickle.dump(model, open('randomforest.sav', 'wb'))
    getPredictions(model, imgTest, maskPath, basePath, outPath)

if __name__=='__main__':

    testSize = 0.1
    basePath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation'
    imagePath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/three/images'
    maskPath  = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/three/masks'
    outPath = '/home/verghese/output/segmentation/mlpixelwise_multi'

    getSegmentation(basePath, imagePath, maskPath, outPath)
