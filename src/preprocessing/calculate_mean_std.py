#!/usr/bin/env python3

import os
import glob
import argparse

import cv2
import numpy as np


def calculate_std_mean(path):

    images = glob.glob(os.path.join(path,'*'))
    image_shape = cv2.imread(images[0]).shape
    channel_num = image_shape[-1]
    channel_values = np.zeros((channel_num))
    channel_values_sq = np.zeros((channel_num))

    pixel_num = len(images)*image_shape[0]*image_shape[1]
    print('total number pixels: {}'.format(pixel_num))

    for path in images:
        print(path)
        image = cv2.imread(path)
        image = (image/255.0).astype('float64')
        channel_values += np.sum(image, axis=(0,1), dtype='float64')

    mean=channel_values/pixel_num
    print("mean:",mean)

    for path in images:
        print(path)
        image = cv2.imread(path)
        image = (image/255.0).astype('float64')
        channel_values_sq += np.sum(np.square(image-mean), axis=(0,1), dtype='float64')

    std=np.sqrt(channel_values_sq/pixel_num, dtype='float64')
    print('mean: {}, std: {}'.format(mean, std))

    return mean, std


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--path', required=True, help='path to image set')
    args = vars(ap.parse_args())

    mean, std = calculate_std_mean(args['path'])

    










