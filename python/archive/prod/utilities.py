import os
import pickle
import random
import shutil
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.preprocessing import image


'''
Copy proportion of files to train and validation
based on the ratio split.
'''
def MoveFiles(baseDir, folder, ratio, trainDest=None, validDest=None, testDest=None, copy=True):

    files = os.listdir(baseDir+'/'+folder)
    random.shuffle(files)
    num = round(ratio*len(files))
    trainFiles = []
    testFiles = []

    for i in range(num):
        f = random.choice(files)
        files.remove(f)
        trainFiles.append(f)

    testNum = round(0.5*len(files))

    for i in range(testNum):
        f = random.choice(files)
        files.remove(f)
        testFiles.append(f)

    validFiles = [f for f in files]

    if copy:
        for f in trainFiles:
            shutil.copy(os.path.join(baseDir, folder, f), trainDest)

        for f in testFiles:
            shutil.copy(os.path.join(baseDir,folder,f), testDest)

        for f in validFiles:
            shutil.copy(os.path.join(baseDir, folder, f), validDest)

    return trainFiles, testFiles, validFiles


'''
Image generator. Returns numpy representation of image and label
'''
def getImages(imgPath, label, targetSize):

    for f in os.listdir(imgPath):
        if 'png' in f:
            img = image.load_img(imgPath+'/'+f, target_size=targetSize)
            imgArr = image.img_to_array(img)
            img.close()
            imgArr = np.expand_dims(imgArr, axis=0)

            yield (imgArr, label)


'''
save model down as Json - separate from weights
'''
def saveModelJson(model, modelName, filePath='models/'):

    model_json = model.to_json()

    with open(filePath+modelName+'.json', 'w') as json_file:
        json_file.write(model_json)

'''
load model from json
'''
def loadModelJson(modelPath):

    with open(modelPath, 'rb') as jsonFile:
        loadedModelJson = jsonFile.read()

    modelJson = model_from_json(loadedModelJson)


    return loadedModelJson

'''
save model down as H5 (weights and model)
'''
def saveModel(model, modelName, filePath='/home/verghese/models'):
    model.save(os.path.join(filePath, modelName + '.h5'))


'''
load model from H5
'''
def loadModelH5(modelPath):

    loadedModel = load_model(modelPath)
    return loadedModel


'''
save history down
'''
def saveHistory(history, name):

    with open('models/'+name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


'''
load save history. Note returns as dictionary
'''
def loadHistory(name):

    with open('models/'+name, 'rb') as f:
        history = pickle.load(f)

    return history


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



def getTrainMetric(history, tMetricName, vMetricName, outPath, modelname):
      
    sns.set_style('dark')
    folder = os.path.join(outPath, modelname)
   
    trainMetric = history.history[tMetricName]
    valMetric = history.history[vMetricName]
    epochs = range(len(trainMetric))
     
    fig = plt.figure(figsize=(8,5))
    sns.lineplot(range(len(trainMetric)), trainMetric, markers=True, dashes=False, label='Training ' + tMetricName)
    sns.lineplot(range(len(trainMetric)), valMetric, markers=True, dashes=False, label='Validation ' + tMetricName)
    plt.title('Training and validation ' + tMetricName)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    
    plt.legend()
    plt.grid()
    plt.show()
    fig.savefig(os.path.join(folder, modelname+tMetricName+'_graph.png'))
    plt.close()

