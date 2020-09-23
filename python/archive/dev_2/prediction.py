import os
import glob
import tensorflow as tf
import numpy as np
import pandas
import matplotlib.pyplot as plt
from evaluation import iouScore, diceCoef


def getPrediction(model, testdataset, num, modelname, batchSize,  outPath):

    folder=os.path.join(outPath, modelname)
    try:
        os.mkdir(folder)
    except Exception as e:
        pass

    diceLst=[]
    iouLst=[]
  
    #for idx, batch in enumerate(testdataset):
     
    for i, data in enumerate(testdataset):
        image = data[0].numpy().astype(np.int16)
        mask = data[1].numpy().astype(np.int16)
        image2 = np.expand_dims(image, axis=0).astype(np.float32)
        mask2 = np.expand_dims(mask, axis=0).astype(np.float32) 
        predProbs = model.predict(image2)
        prediction = (predProbs > 0.5).astype('int16')
        prediction = prediction.astype(np.float32)
        mask2[mask2!=0]=1
        dice = diceCoef(mask2, prediction[:,:,:,0])
        diceLst.append(dice.numpy())
        iou = iouScore(mask2, prediction[:,:,:,0])
        iouLst.append(iou.numpy())

        fig, axs = plt.subplots(1, 3, figsize=(5, 5))
        axs[0].imshow(image, cmap='gray')
        axs[1].imshow(mask*255, cmap='gray')
        axs[2].imshow(prediction[0,:,:,0]*255, cmap='gray')
        #index = i+(idx*batchSize)
        fig.savefig(os.path.join(folder, str(i) + '.png'))
        plt.close()

    avgDice = np.mean(np.array(diceLst))
    avgIOU = np.mean(np.array(iouLst))
    print('Avg dice: {} \n Avg iou {}'.format(avgDice, avgIOU))

    imgscores = pandas.DataFrame({'dice':diceLst, 'iou':iouLst})
    imgscores.to_csv(os.path.join(folder, modelname+'_imgscores.csv'))
    summary = pandas.DataFrame({'dice':[avgDice], 'iou': [avgIOU]})
    summary.to_csv(os.path.join(folder, modelname+'_summary.csv'))
    return avgDice
