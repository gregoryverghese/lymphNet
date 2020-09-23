import os
import glob
import numpy as np
import tensorflow as tf
import argparse


def printProgress(count, total):

    complete = float(count)/total
    print('\r- Progress: {0:.1%}'.format(complete), flush=True)


def wrapInt64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrapBytes(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(imageFiles, maskPath, tfRecordPath):

    numImgs = len(imageFiles)
    print('Number of test images: {}'.format(numImgs))

    with tf.io.TFRecordWriter(tfRecordPath) as writer:
        for i, img in enumerate(imageFiles):
            printProgress(i, numImgs-1)
            print(img) 
            label = os.path.basename(img)[:-4]
            m = os.path.join(maskPath, label + '_masks.png')

            image = tf.keras.preprocessing.image.load_img(img)
            image = tf.keras.preprocessing.image.img_to_array(image, dtype=np.uint8)

            imageDim = tf.shape(image).numpy()
            image = tf.image.encode_png(image)
            print('dimensions', imageDim)

            mask = tf.keras.preprocessing.image.load_img(m)
            mask = tf.keras.preprocessing.image.img_to_array(mask, dtype=np.uint8)

            ##########################################################
            #ToDo: Not sure why values change to rgb (0-255) from binary
            #Need to find the source of problem not use this work around
            #masknew = mask.copy()
            #masknew[masknew!=0]=1
            #masknew=masknew.astype(np.uint16)
            ###########################################################

            maskDim = tf.shape(mask).numpy()
            mask = tf.image.encode_png(mask)

            print('imageDim: {}, maskDim: {}'.format(imageDim, maskDim))
            
            dataMap = {'image': wrapBytes(image),
                        'mask': wrapBytes(mask),
                        'xDim': wrapInt64(imageDim[0]),
                        'yDim': wrapInt64(maskDim[1]),
                        'label': wrapBytes(label.encode('utf-8'))
                        }

            features = tf.train.Features(feature=dataMap)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-op', '--outpath', required=True, help='path to save records to')
    ap.add_argument('-ip', '--imagepath', required=True, help='path where images are saved')
    ap.add_argument('-mp', '--maskpath', required=True, help='path where masks are saved')
    ap.add_argument('-f', '--feature', required=True, help='histological feature of interest')

    args = vars(ap.parse_args())

    imagePath = args['imagepath']
    maskPath = args['maskpath']
    outPath = args['outpath']
    feature = args['feature']

    print('Generating test records for {}'.format(feature))
    
    imageFiles = glob.glob(os.path.join(imagePath, '*'))
    convert(imageFiles, maskPath, os.path.join(outPath, feature+'.tfrecords'))




