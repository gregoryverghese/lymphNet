'''
tfrecord_write.py: write down image and mask data in serialized tfrecord format
'''

import os
import json
import glob
import argparse
import math

import cv2
import numpy as np
import staintools
import tensorflow as tf

__author__= 'Gregory Verghese'
__email__='gregory.verghese@gmail.com'

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

DEBUG = True

#####HR - this is NOT a good place to stain normalise
TARGET='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/norm-targets/14.90610 C L2.11.png'

def stain_normalizer(image):
    image.setflags(write=1)
    target_path=TARGET
    target=cv2.imread(target_path)
    target=cv2.cvtColor(target,cv2.COLOR_BGR2RGB)
    normalizer=staintools.StainNormalizer(method='vahadane')
    normalizer.fit(target)
    try:
        transformed = normalizer.transform(image)
    except Exception as e:
        print('Not-transformed')
        transformed=None
    return transformed


def getShardNumber(images, masks, shardSize=0.25, unit=10**9):
    '''
    calculate the number of shards based on images
    masks and required shard size
    Args:
        images: image paths
        masks: mask paths
        shardsize: memory size of each shard
        unit: gb
    Returns:
        shardNum: number of shards
        imgPerShard: number of images in each shard
    '''
    print("num ims:",len(images))
    print("num masks:",len(masks))
    print(images)
    print(masks)
    maskMem = sum(os.path.getsize(f) for f in masks if os.path.isfile(f))
    imageMem = sum(os.path.getsize(f) for f in images if os.path.isfile(f))
    totalMem = (maskMem+imageMem)/unit
    print('Image memory: {}, Mask memory: {}, Total memory: {}'.format(imageMem, maskMem, totalMem))    
    shardNum = int(np.ceil(totalMem/shardSize))
    imgPerShard = int(np.floor(len(images)/shardNum))

    return shardNum, imgPerShard


def printProgress(count, total):
    '''
    print progress of saving
    Args:
        count: current image number
    total:
        total: total number of images
    '''
    complete = float(count)/total
    print('\r- Progress: {0:.1%}'.format(complete), flush=True)


def wrapInt64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrapFloat(value):
    '''
    convert to tf float
    Returns:

    '''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def wrapBytes(value):
    '''
    convert value to bytes
    Args:
        value: image
    Returns:

    '''
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(imageFiles, maskFiles, tfRecordPath, dim=None):
    '''
    load images and masks and serialize as a tfrecord file
    Args:
        imageFiles: imagefile paths
        maskFiles: maskfile paths
        tfRecordPath: path to save tfrecords
    '''

    numImgs = len(imageFiles)
    check=[]
    with tf.io.TFRecordWriter(tfRecordPath) as writer:
        for i, (img, m) in enumerate(zip(imageFiles, maskFiles)):
            printProgress(i,numImgs)
            imgName = os.path.basename(img)[:-4]
            maskName = os.path.basename(m)[:-10]
            mPath = os.path.dirname(m)
           
            m = os.path.join(mPath, os.path.basename(img)) #[:-4]) + '_mask.png')
            if DEBUG: print(imgName)
            if DEBUG: print(m)
            maskName = os.path.basename(m)
            if not os.path.exists(m):
                if DEBUG: print("mask does not exist")
                check.append(maskName)
                continue
 
            maskName = os.path.basename(m)

            ## in holly-old-branch the two commented out lines were used
            ## added by GV in Nov 22
            image = np.array(tf.keras.preprocessing.image.load_img(img,color_mode='rgb'))
            #image = tf.keras.preprocessing.image.img_to_array(image,dtype=np.uint8)
            #image = stain_normalizer(image)
            dims = image.shape
            image = tf.image.encode_png(image)
            
            mask = tf.keras.preprocessing.image.load_img(m)
            mask = tf.keras.preprocessing.image.img_to_array(mask, dtype=np.uint8)
            mask = tf.image.encode_png(mask)

            data = {
                'image': wrapBytes(image),
                'mask': wrapBytes(mask),
                'imageName': wrapBytes(os.path.basename(img)[:-4].encode('utf-8')),
                'maskName': wrapBytes(os.path.basename(m)[:-4].encode('utf-8')),
                'dims': wrapInt64(dims[0]) 
                }
               
            features = tf.train.Features(feature=data)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)

        print('Number of errors: {}'.format(len(check)))    


def doConversion(imgs, masks, shardNum, num, outPath, outDir):
    '''
    split files into shards for saving down
    Args:
        imgs: list of image tensors
        masks: list of mask image tensors
        shardNum: number of shards
        num: number of images per shard
        outPath: path to save files
        outDir: directory to save files
    '''
    for i in range(0, shardNum):
        shardImgs = imgs[i*num:num*(i+1)]
        shardMasks = masks[i*num:num*(i+1)]
        convert(shardImgs, shardMasks, os.path.join(outPath,outDir,str(i)+'.tfrecords'), dim=None)
    
    if shardNum > 1:
        shardImgs = imgs[i*num:]
        shardMasks = masks[i*num:]
      
    convert(shardImgs, shardMasks, os.path.join(outPath,outDir,str(i)+'.tfrecords'), dim=None)


def getFiles(imagePath, maskPath, outPath, config, shardSize=0.1):
    '''
    gets images paths and split into train, valid and test sets
    Args:
        imagePath: path to image files
        maskPath: path to mask files
        outPath: path to save down files
        config: config file path containig names of test images
    '''
    with open(config) as jsonFile:
        configFile = json.load(jsonFile)

    if DEBUG: print(imagePath)
    if DEBUG: print(maskPath)
    validFiles=configFile['validFiles']
    testFiles = configFile['testFiles']
    print(validFiles)
    print(testFiles)
    print(configFile)
    #imagePaths = glob.glob(os.path.join(imagePath, '*/images/*'))
    imagePaths = glob.glob(os.path.join(imagePath,'*.png'))
    #maskPaths = glob.glob(os.path.join(maskPath, '*/mask/*'))
    maskPaths = glob.glob(os.path.join(maskPath,'*.png'))
    print('Total images: {}, Total masks: {}'.format(len(imagePaths), len(maskPaths)))

    trainImgs = [img for img in imagePaths if not any([v for v in validFiles+testFiles if v in img])]
    trainMasks = [m for m in maskPaths if not any([v for v in validFiles+testFiles if v in m])]
    validImgs = [img for img in imagePaths if any([v for v in validFiles if v in img])]
    validMasks = [m for m in maskPaths if any([v for v in validFiles if v in m])]
    testImgs = [img for img in imagePaths if any([v for v in testFiles if v in img])]
    testMasks = [m for m in maskPaths if any([v for v in testFiles if v in m])]
    
    '''
    #datacheck
    import pandas as pd
    x = pd.DataFrame({'train': trainImgs})
    x.to_csv('check.csv')
    '''
    print('train:{}, valid: {}, test: {}'.format(len(trainImgs), len(validImgs), len(testImgs)))
    print('train:{}, valid: {}, test: {}'.format(len(trainMasks), len(validMasks), len(testMasks)))
     
    trainShardNum, tNum = getShardNumber(trainImgs, trainMasks)
    doConversion(trainImgs, trainMasks, trainShardNum, tNum, outPath, 'train')
    print('Number of train shards: {}'.format(trainShardNum))


    ################# HERE ###################
    validShardNum, vNum = getShardNumber(validImgs, validMasks, shardSize=0.1)
    doConversion(validImgs, validMasks, validShardNum, vNum, outPath, 'validation')
    print('Number of validation shards: {}'.format(validShardNum))

    #testShardNum, tstNum = getShardNumber(validImgs, validMasks, shardSize=0.005)
    #doConversion(validImgs, validMasks, validShardNum, vNum, outPath, 'test')
    testShardNum, tstNum = getShardNumber(testImgs, testMasks,shardSize=0.1)
    doConversion(testImgs, testMasks, testShardNum, tstNum, outPath, 'test')
    print('Number of test shards: {}'.format(testShardNum))
    

def basicConvert(imagePath, maskPath, outPath, shardSize=0.1):
    '''
    gets images paths and convert to tfrecords without splitting into test, validate and train
    necessary for converting files that have previously been extracted from tfrecords and have lost their filenames (Holly Rafique)
    Args:
        imagePath: path to image files
        maskPath: path to mask files
        outPath: path to save down files
    '''

    imagePaths = glob.glob(os.path.join(imagePath, '*'))
    maskPaths = glob.glob(os.path.join(maskPath, '*'))

    print('Total images: {}, Total masks: {}'.format(len(imagePaths), len(maskPaths)))

    allShardNum, shNum = getShardNumber(imagePaths, maskPaths)
    doConversion(imagePaths, maskPaths, allShardNum, shNum, outPath, '')
    print('Number of shards: {}'.format(allShardNum))

def basicConvert(imagePath, maskPath, outPath, shardSize=0.1):
    '''
    gets images paths and convert to tfrecords without splitting into test, validate and train
    necessary for converting files that have previously been extracted from tfrecords and have lost their filenames (Holly Rafique)
    Args:
        imagePath: path to image files
        maskPath: path to mask files
        outPath: path to save down files
    '''

    imagePaths = glob.glob(os.path.join(imagePath, '*'))
    maskPaths = glob.glob(os.path.join(maskPath, '*'))

    print('Total images: {}, Total masks: {}'.format(len(imagePaths), len(maskPaths)))

    allShardNum, shNum = getShardNumber(imagePaths, maskPaths)
    doConversion(imagePaths, maskPaths, allShardNum, shNum, outPath, '')
    print('Number of shards: {}'.format(allShardNum))



if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-fp', '--filepath', required=True, help='path to images')
    ap.add_argument('-mp', '--maskpath', required=True, help='path to mask')
    ap.add_argument('-op', '--outpath', required=True, help='path for tfRecords to be wrriten to')
    ap.add_argument('-cf', '--configfile', help='path to config file')
    args = vars(ap.parse_args())
    os.makedirs(os.path.join(args['outpath'],'train'),exist_ok=True)
    os.makedirs(os.path.join(args['outpath'],'test'),exist_ok=True)
    os.makedirs(os.path.join(args['outpath'],'validation'),exist_ok=True)

    print("im: ",args['filepath'])
    #print("images: ",args.filepath)
    #print("masks: ",args.maskpath)
    #print("config: ",args.configfile)
    #print("out: ",args.outpath)

    #getFiles(args['filepath'], args['maskpath'], args['outpath'], args['configfile'])
    if(args['configfile']):
        print("calling getFiles")
        getFiles(args['filepath'], args['maskpath'], args['outpath'], args['configfile'])
    else:
        print("calling basicConvert")
        basicConvert(args['filepath'], args['maskpath'], args['outpath'])

