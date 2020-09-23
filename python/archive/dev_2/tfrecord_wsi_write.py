import os
import glob
import numpy as np
import tensorflow as tf

def printProgress(count, total):

    complete = float(count)/total
    #print('\r- Progress: {0:.1%}'.format(complete), flush=True)


def wrapInt64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrapBytes(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(imageFiles, maskPath, tfRecordPath):

    numImgs = len(imageFiles)

    with tf.io.TFRecordWriter(tfRecordPath) as writer:
        for i, img in enumerate(imageFiles):
            #printProgress(i, numImgs-1)
            
            label = os.path.basename(img)[:-4]
            m = os.path.join(maskPath, label + '_masks.png')

            image = tf.keras.preprocessing.image.load_img(img)
            image = tf.keras.preprocessing.image.img_to_array(image, dtype=np.uint8)
            image = tf.image.encode_png(image)

            mask = tf.keras.preprocessing.image.load_img(m)
            mask = tf.keras.preprocessing.image.img_to_array(mask, dtype=np.uint8)
            mask = tf.image.encode_png(mask)

            dataMap = {'image': wrapBytes(image),
                        'mask': wrapBytes(mask),
                        'label': wrapBytes(label.encode('utf-8'))
                        }

            features = tf.train.Features(feature=dataMap)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)

outPath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/5x/one/tfrecords/tfrecords_wsi/follicle'
images = glob.glob('output/patches/3/*')
maskPath = 'output/masks/3/follicle'

print(images, flush=True)


for image in images:
    patches = glob.glob(os.path.join(image, '*'))
    print(patches, flush=True)
    imageName = os.path.basename(image)

    maskPath2 = os.path.join(maskPath, imageName)
    convert(patches, maskPath2, os.path.join(outPath, imageName + '.tfrecords'))
