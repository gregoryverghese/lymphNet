import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import datetime
import glob
import os
import pandas as pd
from evaluation import diceCoef, iouScore
import tensorflow.keras.backend as K

models = glob.glob('models2/10x/*')
images = glob.glob('10x/images/sinus/*')
maskpath = '10x/masks/sinus'
images = [i for i in images if 'U_100188_15_B_NA_15_L1' in i]
#images = [i for i in images if '32.90577 C L1.2' not in i]

for m in models:

    model = load_model(m)

    currentTime = datetime.datetime.now().strftime('%H:%M')
    currentDate = str(datetime.date.today())
    modelname = os.path.basename(m)[:-2]
    outPath = os.path.join('results',currentDate, modelname) 
    
    try:
        os.mkdir(os.path.join('results',currentDate))
    except Exception as e:
        print(e)

    try:
        os.mkdir(os.path.join('results',currentDate, modelname))
    except Exception as e:
        print(e)

    names = []
    diceLst = []
    iouLst = []
    
    for img in images:
        
        imgname = os.path.basename(img)[:-4]
        print(imgname) 
        image = cv2.imread(img)
        mask = cv2.imread(os.path.join(maskpath, imgname + '_masks.png'))
        image = tf.cast(image, tf.float32)
        mask = tf.cast(mask, tf.float32)
        print(image.shape, mask.shape)

        image = tf.expand_dims(image, axis=0)
        mask = mask.numpy().astype(np.uint8)
        mask[mask!=0]=1

        probs = model.predict(image)

        prediction = tf.cast((probs> 0.5), tf.float32)
        prediction = prediction.numpy().astype(np.uint8)
        mask = tf.cast(tf.convert_to_tensor(mask), tf.float32)
        mask = tf.expand_dims(mask, axis=0)
        prediction = tf.cast(prediction, tf.float32)

        print('final_dims', K.int_shape(mask), K.int_shape(prediction))
        dice = diceCoef(mask[:,:,:,2:], prediction)
        iou = iouScore(mask[:,:,:,0:1], prediction)
        
        print(dice.numpy())
        prediction = prediction.numpy().astype(np.uint8)
        print('prediction', prediction.shape)

        cv2.imwrite(os.path.join(outPath, imgname + '_pred.png'),prediction[0,:,:,0]*int(255))

        diceLst.append(dice.numpy())
        iouLst.append(iou.numpy())

        names.append(imgname)

    imgscores = pd.DataFrame({'image': names, 'dice':diceLst, 'iou':iouLst})
    imgscores.to_csv(os.path.join(outPath, '_imgscores.csv'))

    avgDice = np.mean(diceLst)
    avgIOU = np.mean(iouLst)

    summary = pd.DataFrame({'dice':[avgDice], 'iou': [avgIOU]})
    summary.to_csv(os.path.join(outPath, '_summary.csv'))
