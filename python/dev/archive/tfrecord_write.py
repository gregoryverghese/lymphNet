import cv2
import json
import os
import numpy as np
import glob
import argparse
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def getShardNumber(images, masks, shardSize=0.1, unit=10**9):

    maskMem = sum(os.path.getsize(f) for f in masks if os.path.isfile(f))
    imageMem = sum(os.path.getsize(f) for f in images if os.path.isfile(f))
    totalMem = (maskMem+imageMem)/unit
    print('Image memory: {}, Mask memory: {}, Total memory: {}'.format(imageMem, maskMem, totalMem))
    
    shardNum = int(np.ceil(totalMem/shardSize))
    imgPerShard = int(np.floor(len(images)/shardNum))

    return shardNum, imgPerShard


def printProgress(count, total):

    complete = float(count)/total
    print('\r- Progress: {0:.1%}'.format(complete), flush=True)


def wrapInt64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrapFloat(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def wrapBytes(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(imageFiles, maskFiles, tfRecordPath, dim=None):

    numImgs = len(imageFiles)
    check=[]
    with tf.io.TFRecordWriter(tfRecordPath) as writer:
        for i, (img, m) in enumerate(zip(imageFiles, maskFiles)):
            printProgress(i,numImgs)
            imgName = os.path.basename(img)[:-4]
            maskName = os.path.basename(m)[:-10]
            print(imgName)
            print(maskName)
            if not imgName==maskName:
                continue
                check.append(imgName)
            
            image = tf.keras.preprocessing.image.load_img(img)
            image = tf.keras.preprocessing.image.img_to_array(image, dtype=np.uint8)
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
    
    print('shardNum', shardNum)
    print('num', num)

    for i in range(0, shardNum):
        shardImgs = imgs[i*num:num*(i+1)]
        shardMasks = masks[i*num:num*(i+1)]
        convert(shardImgs, shardMasks, os.path.join(outPath,outDir,str(i)+'.tfrecords'), dim=None)

    if shardNum > 1:
        shardImgs = imgs[i*num:]
        shardMasks = masks[i*num:]
        convert(shardImgs, shardMasks, os.path.join(outPath,outDir,str(i)+'.tfrecords'), dim=None)


def getFiles(imagePath, maskPath, outPath, config, shardSize=0.1):

    with open(config) as jsonFile:
        configFile = json.load(jsonFile)

    validFiles=configFile['validFiles']
    testFiles = configFile['testFiles']

    imagePaths = glob.glob(os.path.join(imagePath, '*'))
    maskPaths = glob.glob(os.path.join(maskPath, '*'))
    
    print('Total images: {}, Total masks: {}'.format(len(imagePaths), len(maskPaths)))

    trainImgs = [img for img in imagePaths if not any([v for v in validFiles+testFiles if v in img])]
    trainMasks = [m for m in maskPaths if not any([v for v in validFiles+testFiles if v in m])]
    validImgs = [img for img in imagePaths if any([v for v in validFiles if v in img])]
    validMasks = [m for m in maskPaths if any([v for v in validFiles if v in m])]
    testImgs = [img for img in imagePaths if any([v for v in testFiles if v in img])]
    testMasks = [m for m in maskPaths if any([v for v in testFiles if v in m])]

    print('train:{}, valid: {}, test: {}'.format(len(trainImgs), len(validImgs), len(testImgs)))
    print('train:{}, valid: {}, test: {}'.format(len(trainMasks), len(validMasks), len(testMasks)))
     
    trainShardNum, tNum = getShardNumber(trainImgs, trainMasks)
    doConversion(trainImgs, trainMasks, trainShardNum, tNum, outPath, 'train')
    print('Number of train shards: {}'.format(trainShardNum))

    validShardNum, vNum = getShardNumber(validImgs, validMasks)
    doConversion(validImgs, validMasks, validShardNum, vNum, outPath, 'validation')
    print('Number of validation shards: {}'.format(validShardNum))

    testShardNum, tstNum = getShardNumber(testImgs, testMasks)
    doConversion(testImgs, testMasks, testShardNum, tstNum, outPath, 'test')
    print('Number of test shards: {}'.format(testShardNum))
    


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-fp', '--filepath', required=True, help='path to images')
    ap.add_argument('-mp', '--maskpath', required=True, help='path to mask')
    ap.add_argument('-op', '--outpath', required=True, help='path for tfRecords to be wrriten to')
    ap.add_argument('-cf', '--configfile', required=True, help='path to config file')
    args = vars(ap.parse_args())

    getFiles(args['filepath'], args['maskpath'], args['outpath'], args['configfile'])
