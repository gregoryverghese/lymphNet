import os
import cv2
import glob
import utilities
import numpy as np
import pickle
import random
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow import keras
from resunet_multi import Unet
from evaluation import dice_coef_loss, dice_coef
from skimage.transform import resize
from skimage import img_as_bool

PARAMS = {'batchSize':16, 'imageSize': (224, 224, 3), 'nClasses':4, 'shuffle':True}
EPOCH = 10
RATIO = 0.9


class DataGenerator(keras.utils.Sequence):
    def __init__(self, imgIds, maskIds, imagePath, maskPath, batchSize=16, imageSize = (224, 224, 3), nClasses=4, shuffle=False):
        self.imgIds = imgIds
        self.maskIds = maskIds
        self.imagePath = imagePath
        self.maskPath = maskPath
        self.batchSize = batchSize
        self.imageSize = imageSize
        self.nClasses = nClasses
        self.shuffle = shuffle


    '''
    for each image id load the patch and corresponding mask
    '''
    def __load__(self, imgName, maskName):
        print(imgName)
        img = cv2.imread(os.path.join(self.imagePath,imgName))
        img = cv2.resize(img, (self.imageSize[0], self.imageSize[1]))

        mask = cv2.imread(os.path.join(self.maskPath,maskName))
        mask = np.dstack((mask, np.zeros((4000, 4000))))

        mask[:,:,2][mask[:,:,0]==0]=255
        mask = mask.astype(np.bool)
        mask = img_as_bool(resize(mask, (self.imageSize[0], self.imageSize[1])))
        mask = mask.astype('uint8')

        img = img/255.0
        print('shape', img.shape, mask.shape)

        return (img, mask)


    '''
    get the files for each batch (override __getitem__ method)
    '''
    def __getitem__(self, index):

        if(index+1)*self.batchSize > len(self.imgIds):
            self.batchSize = len(self.imgIds) - index*self.batchSize
        
        batchImgs = self.imgIds[self.batchSize*index:self.batchSize*(index+1)]
        batchMasks = self.maskIds[self.batchSize*index:self.batchSize*(index+1)]
        batchfiles = [self.__load__(imgFile, maskFile) for imgFile, maskFile in zip(batchImgs, batchMasks)]
        images, masks = zip(*batchfiles)

        return np.array(list(images)), np.array(list(masks))


    '''
    Return number of steps per batch that are needed (override __len__ method)
    '''
    def __len__(self):
        return int(np.ceil(len(self.imgIds)/self.batchSize))


'''
save model down
'''
def saveModel(model, modelName, filePath='/home/verghese/models/'):

    model.save(filePath+modelName+'_seg_1.h5')


'''
save history down
'''
def saveHistory(history, name):

    with open('/home/verghese/models/'+name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


'''
load save history. Note returns as dictionary
'''
def loadHistory(name):

    with open('/home/verghese/models/'+name, 'rb') as f:
        history = pickle.load(f)

    return history


'''
compile keras model
'''
#def compileModel(model, loss=loss, optimizer=optimizer, metrics=['acc']):
    #model.compile(loss=loss, optimizer=optimizer,  metrics=metrics)
    #return model


'''
fit model to training data using train generator
'''
def fitModel(model, trainGenerator, validationGenerator, callbacks, epochs=10):

    history = model.fit_generator(
    trainGenerator,
    steps_per_epoch=trainGenerator.samples//trainGenerator.batch_size,
    epochs=epochs,
    validation_data=validationGenerator,
    validation_steps=validationGenerator.samples//validationGenerator.batch_size,
    verbose=1)

    return history, model


def getPred(model, validGenerator, validIds):

    steps = len(validIds)//validGenerator.batchSize

    for i in range(0, steps):
        x, y = validGenerator.__getitem__(i)
        y[y==1]=255
        masks.append(y)
        yPred = model.predict(x)
        yPred = np.argmax(yPred, axis=3)

        for img in yPred:
                x, y = validGenerator.__getitem__(i)
                y[y==1]=255
                masks.append(y)
                yPred = model.predict(x)
                yPred = np.argmax(yPred, axis=3)


def weighted_categorical_crossentropy(weights):

    def wcce(y_true, y_pred):
        Kweights = K.constant(weights)
        if not K.is_tensor(y_pred):
            y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)
    return wcce


def getSegmentation(basePath, params, imgFolder='two/images', maskFolder='two/masks'):

    imagePath = os.path.join(basePath, imgFolder)
    maskPath = os.path.join(basePath, maskFolder)
    filePaths = glob.glob(os.path.join(imagePath, '*'))

    ids = [os.path.basename(fp) for fp in filePaths]
    imgIds = glob.glob(os.path.join(imagePath, '*'))
    imgIds = [os.path.basename(f) for f in imgIds]
    maskIds = glob.glob(os.path.join(maskPath, '*'))
    maskIds = [os.path.basename(f) for f in maskIds]
    trainNum = round(RATIO*len(ids))
    validNum = np.floor((len(ids) - trainNum))

    trainIds = imgIds[:trainNum]
    validIds = imgIds[trainNum:]
    #testIds = imgIds[(trainNum+validNum):]
    trainMasks = maskIds[:trainNum]
    validMasks = maskIds[trainNum:]
    #testMasks = maskIds[(trainNum+validNum):]

    trainGenerator = DataGenerator(trainIds, trainMasks, imagePath, maskPath, **params)
    validGenerator = DataGenerator(validIds, validMasks, imagePath, maskPath)
    #testGenerator = DataGenerator(testIds, validMasks, imagePath, maskPath)

    trainSteps = len(trainIds)//trainGenerator.batchSize
    validSteps = len(validIds)//validGenerator.batchSize

    labels = []
    


   # for i in range(trainSteps):
       # _, m = trainGenerator.__getitem__(i)
        #mask = np.argmax(m, axis=3)
        #labels.append(mask.reshape(-1))
       

    #labels = [l.tolist() for l in labels]
    #labels = itertools.chain(*labels)
    #classWeights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
    classWeights= np.array([2.45422474, 4.0880802 , 1.52711433, 0.37131986])
    unet = Unet(224, nOutput=params['nClasses'], activation='relu', finalActivation='softmax', padding='same')
    model = unet.ResUnet()
    adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss=weighted_categorical_crossentropy(classWeights), metrics=[dice_coef])

    trainSteps = len(trainIds)//trainGenerator.batchSize
    validSteps = len(validIds)//validGenerator.batchSize

    history = model.fit_generator(trainGenerator,
                    validation_data=validGenerator,
                    steps_per_epoch=trainSteps,
                    validation_steps=validSteps,
                    verbose=1,
                    epochs=EPOCH)

    saveModel(model, 'unet_bi_weighted')
    saveHistory(history, 'unet_bi_weighted')

    #getPred(model, validGenerator, validIds)


if __name__ == '__main__':
    BASEPATH = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation'
    getSegmentation(BASEPATH, PARAMS)
