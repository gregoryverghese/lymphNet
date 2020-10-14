import cv2
import os
import numpy as np
import pickle
import tensorflow as tf
#from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import random
#from custom_datagenerator import DataGenerator
from resunet import Unet
import utilities
import glob
from skimage.util import img_as_float
from skimage import img_as_bool
from skimage.transform import resize


#BASEPATH= 'test3/segmentation'
PARAMS = {'batchSize':16, 'imageSize': (224, 224, 3), 'nClasses':2, 'shuffle':True}
EPOCH = 10
RATIO = 0.8


class DataGenerator(keras.utils.Sequence):
    def __init__(self, ids, path, batchSize=3, imageSize = (224, 224, 3), nClasses=2, shuffle=False):
        self.path = path
        self.batchSize = batchSize
        self.imageSize = imageSize
        self.nClasses = nClasses
        self.shuffle = shuffle
        self.ids = ids
        #self.onEpochEnd()


    def getIds(self):

        ids = os.listdir(self.path)
        ids = [i for i in ids if '.png' in i]
        round(len(ids)*ratio)


    '''
    for each image id load the patch and corresponding mask
    '''
    def __load__(self, imgName):

        imagePath = os.path.join(self.path,  'images', imgName)
        maskName =  imgName[:-4]+'mask.png'
        maskPath = os.path.join(self.path, 'masks', maskName)
        
        print('maskName', maskPath)

        img = cv2.imread(imagePath)

        print(img)
        img = cv2.resize(img, (self.imageSize[0], self.imageSize[1]))

        mask = cv2.imread(maskPath, 0)
        #mask = cv2.resize(mask, (self.imageSize[0], self.imageSize[1]))
        mask = mask.astype(np.bool)
        mask = img_as_bool(resize(mask, (self.imageSize[0], self.imageSize[1])))
        mask = mask.astype('uint8')
        mask[mask==1]=255
        mask = np.expand_dims(mask, axis=-1)

        img = img/255.0
        mask = mask/255.0

        return (img, mask)


    '''
    get the files for each batch (override __getitem__ method)
    '''
    def __getitem__(self, index):

        if(index+1)*self.batchSize > len(self.ids):
            self.batchSize = len(self.ids) - index*self.batchSize

        batchFileNames = self.ids[self.batchSize*index:self.batchSize*(index+1)]

        batchfiles = [self.__load__(f) for f in batchFileNames]

        images, masks = zip(*batchfiles)

        return np.array(list(images)), np.array(list(masks))


    '''
    Return number of steps per batch that are needed (override __len__ method)
    '''
    def __len__(self):
        return int(np.ceil(len(self.ids)/self.batchSize))


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
def compileModel(model, loss='binary_crossentropy', optimizer='adam', metrics=['acc']):
    model.compile(loss=loss, optimizer=optimizer,  metrics=metrics)
    return model


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


def getPred(model, testGenerator, testFiles):

    for i in range(len(testFiles)):
        x, y = testGenerator.__getitem__(i)
        result = model.predict(x)
        result = result > 0.4


def getSegmentation():

    BASEPATH = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation'
    imagePath = os.path.join(BASEPATH, 'images')
    #trainFiles, testFiles, validFiles = utilities.MoveFiles(BASEDIR, 'images', RATIO, copy=False)

    filePaths = glob.glob(os.path.join(imagePath, '*'))
    print(len(filePaths))
    ids = [os.path.basename(fp) for fp in filePaths]
    print(len(ids))
    trainNum = round(RATIO*len(ids))
    validNum = round((len(ids) - trainNum)*0.5)
    trainIds = ids[:trainNum]
    validIds = ids[trainNum:(trainNum+validNum)]
    testIds = ids[(trainNum+validNum):]

    print('BASEPATH', BASEPATH)
    print('NUMBER OF THINGS', len(trainIds), len(validIds), len(testIds))

    trainGenerator = DataGenerator(trainIds, BASEPATH , **PARAMS)
    validGenerator = DataGenerator(validIds, BASEPATH, **PARAMS)
    testGenerator = DataGenerator(testIds, BASEPATH, **PARAMS)

    unet = Unet(224)
    model = unet.ResUnet()
    #adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    trainSteps = len(trainIds)//trainGenerator.batchSize
    validSteps = len(validIds)//validGenerator.batchSize

    history = model.fit_generator(trainGenerator,
                    validation_data=validGenerator,
                    steps_per_epoch=trainSteps,
                    validation_steps=validSteps,
                    verbose=1,
                    epochs=EPOCH)
   
    #for i in range(len(model)):
    saveModel(model, 'unet')
    saveHistory(history, 'unet')


   # getPred(model, testGenerator, testIds)

if __name__ == '__main__':
    getSegmentation()
