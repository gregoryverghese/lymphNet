import os
import cv2
import glob
import utilities
import numpy as np
import pickle
import random
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.utils import class_weight
from tensorflow import keras
from resunet_multi import Unet
from evaluation import dice_coef_loss, dice_coef
from skimage.transform import resize
from skimage import img_as_bool
from keras import backend as K

PARAMS = {'batchSize':16, 'imageSize': (224, 224, 3), 'nClasses':4, 'shuffle':True}
EPOCH = 30
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
        img = cv2.imread(os.path.join(self.imagePath,imgName))
        img = cv2.resize(img, (self.imageSize[0], self.imageSize[1]))
        mask = cv2.imread(os.path.join(self.maskPath,maskName))

        mask = img_as_bool(resize(mask, (self.imageSize[0], self.imageSize[1])))
        mask = np.dstack((mask, np.zeros((224, 224))))
        mask[:,:,3][mask[:,:,0]==0]=True
        mask = mask.astype('uint8')

        weights= np.array([8.368834761499912, 11.974269329261341, 6.097782248890891, 1.0])

        for i in range(len(mask)):
            for j in range(len(mask[0])):
                mask[i][j][mask[i,j,:]==1]=weights[mask[i,j,:]==1]

        img = img/255.0
        mask = tf.cast(mask, tf.float32)

        print(mask.shape)
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


def weighted_categorical_crossentropy_fcn_loss(y_true, y_pred):
    # y_true is a matrix of weight-hot vectors (like 1-hot, but they have weights instead of 1s)
    y_true_mask = K.clip(y_true, 0.0, 1.0)  # [0 0 W 0] -> [0 0 1 0] where W >= 1.
    cce = categorical_crossentropy(y_pred, y_true_mask)  # one dim less (each 1hot vector -> float number)
    y_true_weights_maxed = K.max(y_true, axis=-1)  # [0 120 0 0] -> 120 - get weight for each weight-hot vector
    wcce = cce * y_true_weights_maxed
    return K.sum(wcce)


def getSegmentation(basePath, params, imgFolder='three/images', maskFolder='three/masks'):

    imagePath = os.path.join(basePath, imgFolder)
    maskPath = os.path.join(basePath, maskFolder)
    filePaths = glob.glob(os.path.join(imagePath, '*'))

    ids = [os.path.basename(fp) for fp in filePaths]
    imgIds = glob.glob(os.path.join(imagePath, '*'))
    imgIds = [os.path.basename(f) for f in imgIds][:100]
    maskIds = glob.glob(os.path.join(maskPath, '*'))
    maskIds = [os.path.basename(f) for f in maskIds][:100]
    trainNum = round(RATIO*len(imgIds))
    validNum = np.floor((len(imgIds) - trainNum))

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

   # for i in range(trainSteps):
       # _, m = trainGenerator.__getitem__(i)
        #mask = np.argmax(m, axis=3)
        #labels.append(mask.reshape(-1))

    #labels = [l.tolist() for l in labels]
    #labels = itertools.chain(*labels)
    #classWeights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)

    unet = Unet(224, nOutput=params['nClasses'], activation='relu', finalActivation='softmax', padding='same')
    model = unet.ResUnet()
    adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss=weighted_categorical_crossentropy_fcn_loss, metrics=[dice_coef])

    trainSteps = len(trainIds)//trainGenerator.batchSize
    validSteps = len(validIds)//validGenerator.batchSize

    history = model.fit_generator(trainGenerator,
                    validation_data=validGenerator,
                    steps_per_epoch=trainSteps,
                    validation_steps=validSteps,
                    verbose=1,
                    epochs=EPOCH)

    saveModel(model, 'unet_multi_weighted_1')
    saveHistory(history, 'unet_multi_weighted_1')

    #getPred(model, validGenerator, validIds)


if __name__ == '__main__':
    BASEPATH = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation'
    getSegmentation(BASEPATH, PARAMS)
