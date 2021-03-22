import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import glob
import os
import matplotlib.pyplot as plt
from scripts.python.utilities.evaluation import diceCoef, iouScore

modelpath='scripts/models/multiatten_germ_10x_adam_weightedBinaryCrossEntropy_FRC_data_1024_32_07:49.h5'


model=load_model(modelpath)


imagepath='images/10x/*'
maskpath='masks/10x/*'

images = glob.glob(imagepath)
masks = glob.glob(maskpath)
dicelst=[]

for i in images:
    image = cv2.imread(i)
    name = os.path.basename(i)[:-4]
    if 'U_100188_15_B_NA_15_L1' in name:
        continue
    m = [p for p in masks if name in p][0]
    mask=cv2.imread(m)
    image=tf.cast(tf.expand_dims(image,axis=0),tf.float32)
    probs=model.predict(image)
    preds = tf.cast(probs>0.95,tf.float32)
    preds=preds.numpy()                                           
    cv2.imwrite(name + '.png',preds[:,0,0,:])

    mask=tf.cast(tf.expand_dims(mask,axis=0),tf.float32)
    dice=diceCoef(preds,mask[:,:,:,0:1])
    dicelst.append(dice)

df=pd.DataFrame({'dice':dicelst})
df.to_csv('dice.csv')
