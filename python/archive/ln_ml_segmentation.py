import numpy as np
import pandas as pd
import cv2
from skimage.filters import sobel, prewitt, scharr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import itertools
import glob
import shutil
import os
from image_feature_engineering import FeatureEngineering, GaborFilters
from skimage.transform import resize
from skimage import img_as_bool
import pickle

class dataGenerator():
    def __init__(self, basePath, imagePath, maskPath):
        print('gregeg', imagePath)
        self.basePath = basePath
        self.imagePath = imagePath
        self.maskPath = maskPath



    def imageGenerator(self):
        print(self.imagePath)
        print('greg', os.path.join(self.imagePath, '*'))
        classes = glob.glob(os.path.join(self.imagePath, '*'))

        imageFiles = [glob.glob(os.path.join(c, '*')) for c in classes]
        imageFiles = list(itertools.chain(*imageFiles))

        print('The length is', len(imageFiles))
        for f in imageFiles:
            shutil.copy(f, os.path.join(self.basePath, 'images'))


    def maskGenerator(self):

        classes = glob.glob(os.path.join(self.maskPath, '*'))
        step = np.floor(255/len(classes))
        numClass = [(yield c+1) for c in range(len(classes))]

        for c in classes:
            maskFiles = glob.glob(os.path.join(c, '*'))
            num = next(numClass)
            for m in maskFiles:

                mask = cv2.imread(m)
                mask[mask==255]=num*step
                name = m[len(c)+1:]

                print('here we go', os.path.join(self.basePath, 'masks', name))
                cv2.imwrite(os.path.join(self.basePath, 'masks', name), mask)



def getImageMask(filePath, basePath, imgSize=224):
    #print('THIS', basePath)
    img = cv2.imread(filePath)
    img = cv2.resize(img, (imgSize, imgSize))
    maskPath = os.path.join(basePath, 'masks')
    #print('mask', maskPath)
    #print('file', filePath)
    #print('reduced file', filePath[76:-4])
    fileName = filePath[77:-4]
    mFile = os.path.join(maskPath,fileName)+'mask.png'
    #print('grgr', mFile)
    
    mask = cv2.imread(mFile, 0)
    
    mask = mask.astype(np.bool)
    mask = img_as_bool(resize(mask,  (224, 224)))
    mask = mask.astype('uint8')

    return img, mask


def featureExtractor(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fe = FeatureEngineering(img)

    features = GaborFilters(img).getGabors()
    features.append(fe.getCanny(img))
    features.append(fe.getRoberts(img))
    features.append(fe.getSobel(img))
    features.append(fe.getGaussian(img, 3))
    features.append(fe.getGaussian(img, 5))
    features.append(fe.getGaussian(img, 7))
    features.append(fe.getScharr(img))
    features.append(fe.getPrewitt(img))
    features.append(fe.getMedium(img, size=3))
    features.append(fe.getMedium(img, size=3))

    return features


def getFeatures(imgFiles, basePath, imgSize=224):

    imgs = []
    masks = []
    fAll = [np.zeros(imgSize*imgSize)]*42
    fAll = np.transpose(fAll)
    labels = np.zeros(imgSize*imgSize)
    labels = np.transpose(labels)
    labels = labels.reshape(-1, 1)
    i=0
    for f in imgFiles:
        print(i, flush=True)
        img, mask = getImageMask(f, basePath, imgSize=224)
        imgs.append(img)
        masks.append(mask)
        
        features = featureExtractor(img)
        f1 = np.transpose(features)
        fAll = np.vstack((fAll, f1))
        mask = mask.reshape(-1, 1)
        labels = np.vstack((labels, mask))
        
        i=i+1
    
    return fAll, labels, imgs, masks


def trainClassifier(X, y):

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    return model


def savePredImage(yPred, imgTest, testImgs, testMask, imgSize=224):

    segments = [yPred[i:i+(imgSize*imgSize)] for i in range(0, len(yPred), (imgSize*imgSize))]
    segments = list(map(lambda x: x.reshape(imgSize, imgSize), segments))

    for i in range(len(segments)):
        img = segments[i]
        img[img==29]=255

        cv2.imwrite(outDir+'test/'+str(i)+'_pred.png', img)
        cv2.imwrite(outDir+'test/'+str(i)+'_img.png', testImgs[i])
        cv2.imwrite(outDir+'test/'+str(i)+'_mask.png', testMask[i])

    return img


def getPredictions(model, testFiles, basePath, outPath):
    pixelPredictions = []
    pixelLabels = []
    for file in testFiles:

        img, mask = getImageMask(file, basePath, imgSize=224)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        f = featureExtractor(img)
        f = np.array(f)
        f = np.transpose(f)

        pred = model.predict(f)
        maskPred = pred.reshape((224, 224))

        pixelPredictions.append(pred)
        pixelLabels.append(mask.reshape(-1))
        
        maskPred = np.uint8(maskPred)
        contours, hierarchy = cv2.findContours(maskPred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        maxContour = max(contours, key=cv2.contourArea)
        imgPred = cv2.drawContours(img.copy(), maxContour, -1, (0, 255, 0), 3)

        maskContour, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        maxMaskContour = max(maskContour, key=cv2.contourArea)
        imgTrue = cv2.drawContours(img, maxMaskContour, -1, (255, 0, 0), 3)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        maskPred = cv2.cvtColor(maskPred, cv2.COLOR_GRAY2BGR)
        combined = np.concatenate((imgTrue, mask, maskPred, imgPred), axis=1)
        # print(outPath)
        print('outpath', os.path.join(outPath,file[77:-4]) + 'annot.png')
        cv2.imwrite(os.path.join(outPath,file[77:-4]) + 'annot.png', combined)
       
       

    pixelPredictions = np.hstack(pixelPredictions)
    pixelLabels = np.hstack(pixelLabels)
    results = pd.DataFrame({'Predictions': pixelPredictions, 'True': pixelLabels})
    results.to_csv(os.path.join(outPath, 'pixel_results.csv'))

def getSegmentation(basePath, imagePath, maskPath, outPath):

    classPaths = glob.glob(basePath+'/images/*')

   # if len(classPaths)<2:
       # print('image path', imagePath)
        #gen = dataGenerator(basePath, imagePath, maskPath)
       # gen.imageGenerator()
       # gen.maskGenerator()

    print('hehehhehehe')
    imgFiles  = glob.glob(os.path.join(basePath, 'images', '*'))
    imgFiles = imgFiles

    #print(imgFiles)
    imgTrain, imgTest = train_test_split(imgFiles, test_size=0.3, random_state=20)

    trainFeatures, trainLabels, trainImgs, trainMask = getFeatures(imgTrain, basePath)
    model = trainClassifier(trainFeatures, trainLabels)
    
    getPredictions(model, imgTest, basePath, outPath)
     

    with open('randForest', 'wb') as f:
    	pickle.dump(model, f)

    #testFeatures, testLabels, testImgs, testMask = getFeatures(imgTest)
    #yPred = model.predict(testFeatures)
    #savePredImage(yPred, imgTest, testImgs, testMask, imgSize=224)

    #return model, imgTest


if __name__=='__main__':
    basePath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation'
    imagePath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/data/images'
    maskPath  = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/data/masks'
    outPath = '/home/verghese/output/segmentation/mlpixelwise'

    #basePath = 'SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation'
    #imagePath = 'SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/data/images'
    #maskPath  = 'SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/data/masks'
    #outPath = 'SAN/output'

    getSegmentation(basePath, imagePath, maskPath, outPath)
