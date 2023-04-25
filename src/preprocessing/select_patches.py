#!/usr/bin/env python3

import os
import glob
import cv2
import numpy as np

path = "/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/patches/images/v2/10x-1024-512/images"
output_path = "/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/patches/images/v2/10x-1024-512"

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
green_imgs = []
partial_imgs = []

for img_path in images:
    image = cv2.imread(img_path)        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_mean, img_std = cv2.meanStdDev(image)
    img_mean = img_mean.reshape((3,))
    img_std = img_std.reshape((3,))
    
    img_mean[1],img_mean[2] = img_mean[2],img_mean[1]
    #swap colours
    
    if(sum(img_std)/3<=threshold_white):
        white_imgs.append(img_path)
    else:
        #check for green patches
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
        imagemask = cv2.inRange(image, lower_bound, upper_bound)
        imagemask = imagemask/255

        
        if sum(sum(imagemask))>10:
            green_imgs.append(img_path)
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


