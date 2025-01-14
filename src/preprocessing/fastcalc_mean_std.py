#!/usr/bin/env python3

"""
This script provides functions to calculate the mean and standard deviation of pixel values for a set of images.

It includes parallel processing for performance optimization and supports two methods for calculation: a fast method using threading
and an older sequential method.

Author: Holly Rafique
Email: holly.rafique@kcl.ac.uk
"""


import os
import glob
import argparse

import cv2
import numpy as np

from concurrent.futures import ThreadPoolExecutor


def process_image(path: str) -> tuple[int, np.ndarray, np.ndarray]:
    """Process an image to calculate pixel count, sum of pixel values, and sum of squared pixel values.

    Args:
        path (str): Path to the image file.

    Returns:
        tuple[int, np.ndarray, np.ndarray]: A tuple containing the pixel count, sum of pixel values per channel,
        and sum of squared pixel values per channel. Returns zeros if the image fails to load.
    """
    print(f"Input path: {path}")
    image = cv2.imread(path)
    if image is not None:
        image = (image / 255.0).astype('float32')
        # Calculate pixel counts, sum of pixel values, and sum of squared pixel values
        pixel_count = image.shape[0] * image.shape[1]
        pixel_sum = np.sum(image, axis=(0, 1))
        pixel_sum_sq = np.sum(np.square(image), axis=(0, 1))
        return pixel_count, pixel_sum, pixel_sum_sq

    print("Empty image so returning zeros")
    return 0, np.zeros(3), np.zeros(3)  # Return zeros for images that failed to load

def calculate_stats(path: str, num_workers: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Calculate mean and standard deviation of pixel values for all images in a directory.
       This is the faster method with parallelization

    Args:
        path (str): Path to the directory containing image files.
        num_workers (int, optional): Number of worker threads for parallel processing. Defaults to 8.

    Returns:
        tuple[np.ndarray, np.ndarray]: Mean and standard deviation of pixel values per channel.
    """
    total_pixels = 0
    total_sum = np.zeros(3, dtype='float32')
    total_sum_sq = np.zeros(3, dtype='float32')

    images = glob.glob(os.path.join(path,'*.png'))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_image, images)
        
    for count, sum_, sum_sq in results:
        total_pixels += count
        total_sum += sum_
        total_sum_sq += sum_sq

    # Calculate mean and standard deviation
    mean = total_sum / total_pixels
    variance = (total_sum_sq / total_pixels) - np.square(mean)
    std_deviation = np.sqrt(variance)

    return mean, std_deviation



def calculate_std_mean(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Calculate mean and standard deviation of pixel values using a sequential method.
       This is the slower, older method.

    Args:
        path (str): Path to the directory containing image files.

    Returns:
        tuple[np.ndarray, np.ndarray]: Mean and standard deviation of pixel values per channel.
    """
    images = glob.glob(os.path.join(path,'*.png'))
    image_shape = cv2.imread(images[0]).shape
    channel_num = image_shape[-1]
    channel_values = np.zeros((channel_num))
    channel_values_sq = np.zeros((channel_num))

    pixel_num = len(images)*image_shape[0]*image_shape[1]

    for path in images:
        print(path)
        image = cv2.imread(path)
        image = (image/255.0).astype('float32')
        channel_values += np.sum(image, axis=(0,1))

    mean=channel_values/pixel_num

    for path in images:
        image = cv2.imread(path)
        image = (image/255.0).astype('float32')
        channel_values_sq += np.sum(np.square(image-mean), axis=(0,1))

    std=np.sqrt(channel_values_sq/pixel_num)
    print('mean: {}, std: {}'.format(mean, std))

    return mean, std


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--path', required=True, help='path to image set')
    ap.add_argument('-old', '--old_method', action='store_true')
    args = vars(ap.parse_args())
    #args=ap.parse_args()

    #if args.old_method
    if args['old_method']:
        print("old method")
        mean, std = calculate_std_mean(args['path']) #args.path
        print("mean:",mean)
        print("std:",std)
    else:
        print("new method")
        mean_values, std_values = calculate_stats(args['path']) #args.path
        print("mean: ",mean_values)
        print("std: ",std_values)

    print("done")

    










