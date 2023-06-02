#!/usr/bin/env python3

import os
import glob
import cv2
import numpy as np

path = "/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/patches/10x/10x-1024-512/images"
output_path = "/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/patches/10x/10x-1024-512"

def get_percentage_overexposed_rainbows(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image from RGB to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold the value channel to identify over-exposed areas
    #V [0.255]
    v_thresh = 165  # Adjust this threshold to fine-tune the detection 240
    mask_v = cv2.inRange(hsv[:, :, 2], v_thresh, 225)

    # Threshold the hue channel to identify rainbow areas
    #hue [0,179] red is 0-10 and 160-180
    # blue is 110-130
    hue_min = 10  # Adjust this threshold to fine-tune the detection
    hue_max = 140
    mask_h = cv2.inRange(hsv[:, :, 0], hue_min, hue_max)

    # Combine the two masks to identify over-exposed rainbows
    mask = cv2.bitwise_and(mask_v, mask_h)

    # Find contours of the detected areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   # Assume contours is a list of contours obtained from an image
    # Initialize a blank image with the same size as the original image
    img_blank = np.zeros_like(img)
    
    total_area = img.shape[0] * img.shape[1]
    contour_area = np.sum(mask)/255
    percent_within_contours = (contour_area / total_area) * 100

    return percent_within_contours



ci = 1

#xs90 = np.percentile(df.loc['under5']['avgstd'],95)
threshold_white = 5.6

light_m_th = 220 
light_s_th = 15
dark_m_th = [220,210,205] 
dark_s_th = [40,50,60]
print(light_s_th)
print(light_m_th)
print(dark_s_th)
print(dark_m_th)

#defining the lower bounds and upper bounds
lower_bound = np.array([30, 60, 130])
upper_bound = np.array([80,255,255])


images = glob.glob(os.path.join(path,'*'))


white_imgs = []
black_imgs = []
green_imgs = []
partial_imgs = []
swirl_imgs = []

for img_path in images:
    print(img_path)
    #swirl_pct = get_percentage_overexposed_rainbows(img_path)
    image = cv2.imread(img_path)        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_mean, img_std = cv2.meanStdDev(image)
    img_mean = img_mean.reshape((3,))
    img_std = img_std.reshape((3,))
    
    img_mean[1],img_mean[2] = img_mean[2],img_mean[1]
    #swap colours
    
    if(sum(img_std)/3<=threshold_white):
        white_imgs.append(img_path)
    elif img_std[0]>80 and img_std[1]>80 and img_std[2]>80:
        black_imgs.append(img_path)
    else:
        #check for green patches
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
        img_mean_hsv, img_std_hsv = cv2.meanStdDev(image)        

        imagemask = cv2.inRange(image, lower_bound, upper_bound)
        imagemask = imagemask/255

        
        if sum(sum(imagemask))>10:
            green_imgs.append(img_path)
        #elif swirl_pct > 20:
        #    swirl_imgs.append(img_path)
        elif img_mean_hsv[2] > 200:
            if img_mean_hsv[1] < 40:
                partial_imgs.append(img_path)
            elif img_mean_hsv[1] < 55:
                if img_std_hsv[1] > 40:
                    partial_imgs.append(img_path)
        else:
            if (sum(img_std)/3.0) < light_s_th and (img_mean[ci] > light_m_th):
                partial_imgs.append(img_path)
            elif img_std[ci] > dark_s_th[0]:
                if img_std[ci] <=dark_s_th[1]: 
                    if img_mean[ci]>=dark_m_th[0]:
                        partial_imgs.append(img_path)
                elif img_std[ci] <=dark_s_th[2]:
                    if img_mean[ci]>=dark_m_th[1]:
                        partial_imgs.append(img_path)
                elif img_std[ci] > dark_s_th[2]: 
                    if img_mean[ci]>=dark_m_th[2]:
                        partial_imgs.append(img_path)

if(white_imgs):    
    f = open(os.path.join(output_path,'exclude_white.txt'), 'w')
    f.writelines('\n'.join(white_imgs))
    f.write("\n")
    f.close()

if(black_imgs):    
    f = open(os.path.join(output_path,'exclude_black.txt'), 'w')
    f.writelines('\n'.join(black_imgs))
    f.write("\n")
    f.close()

if(green_imgs):
    f = open(os.path.join(output_path,'exclude_green.txt'), 'w')
    f.writelines('\n'.join(green_imgs))
    f.write("\n")
    f.close()

if(partial_imgs):
    f = open(os.path.join(output_path,'exclude_partial.txt'), 'w')
    f.writelines('\n'.join(partial_imgs))
    f.write("\n")
    f.close()

if(swirl_imgs):
    f = open(os.path.join(output_path,'exclude_swirl.txt'), 'w')
    f.writelines('\n'.join(swirl_imgs))
    f.write("\n")
    f.close()



