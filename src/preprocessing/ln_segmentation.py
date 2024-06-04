"""
" ln_segmentation.ipynb
" author: Holly Rafique
" date: 23/02/2024
" based on code by Dina 
" 
" Generates masks to identify lymph node tissue within Whole Slide Images
"
" input: png files of H&E stained WSIs
" output: png masks in the same dimension as the input file
"""

#Dina's model requires v2.14
#Ananya is training on v2.15
# tensorflow version is critical to functioning
#!pip uninstall tensorflow -y
#!pip install tensorflow==2.14

import os
import glob
import shutil
import cv2
import numpy as np

# Import required libraries
import matplotlib.pyplot as plt
#import cv2
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.losses import Loss
from tensorflow.keras.utils import register_keras_serializable

from tensorflow.keras.models import load_model

import openslide
from PIL import UnidentifiedImageError
from openslide import OpenSlideError

print("TensorFlow version:", tf.__version__)



# Define loss function and store in keras class
@register_keras_serializable()
# Inherit class from tf.keras.losses.Loss
class DiceLoss(tf.keras.losses.Loss):
    # Initialise the loss function parameters
    def __init__(self, smooth=1, gama=2):
        super(DiceLoss, self).__init__()
        # Name loss function
        self.name = 'NDL'
        # Apply smoothing factor to avois division by zero
        self.smooth = smooth
        # Exponent valye for gama in denominator
        self.gama = gama

    # Call methds to compute the loss
    def call(self, y_true, y_pred):
        # Convert to float32
        y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)

        # Numerator of the dice loss formula
        nominator = 2 * tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        # Denominator of the dice loss formulae
        denominator = tf.reduce_sum(y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        # Compute the final dice loss result
        result = 1 - tf.divide(nominator, denominator)
        return result


from pathlib import WindowsPath
# Define function to generate binary mask from model prediction
def create_binary_mask(pred_masks, threshold=0.5):
    #argmax Ananya check with Dina
    #tf.argmax(pred_masks,axis=-1)
    # Apply threshold to get binary mask
    print("thresh: ", threshold)

    binary_masks = tf.where(pred_masks > threshold, 1, 0)
    return tf.cast(binary_masks, tf.uint8)

def extract_basename(tensor_path):
    # Split the path by the directory separator
    parts = tf.strings.split(tensor_path, os.path.sep)
    # The last part will be the basename
    basename = parts[-1]
    return basename

########### MAP FUNCTIONS ####################################################
# file_name:  We need to save the file_name so that we can use the same name
#             for saving the masks
#
# dims:       Need to save the original dimensions of the image
#             so that we can resize the resulting mask to match
#

# Define the function to load image and corresponding maks with tensorflow
def load_image(image_path):
    # Read the image path as binary data
    print(image_path)
    image = tf.io.read_file(image_path)
		# Decode the binary image data and specify 3 channels for coloured RGB image
    image = tf.io.decode_png(image,channels=3)

    dims = tf.shape(image)
    fname = extract_basename(image_path)
    return {'file_name': fname, 'image': image, 'dims': dims}

# Define RESIZE function - for feeding to NN
def resize(fname, img, dims, img_size=256):
    # Resize the image with tf.resize
    img = tf.image.resize(img, (img_size, img_size), method="nearest")

    return {'file_name': fname, 'image': img, 'dims': dims}


# Define NORMALIZE function
def normalize(fname, img, dims):
   # Normalize input image to values 0-1
   img = tf.cast(img, tf.float32) / 255.0
   return {'file_name': fname, 'image': img, 'dims': dims}

###############################################################################
### PREPROCESS
###############################################################################
## Reads in all WSIs fro input_folders, gets png thumbnails, 
## writes pngs to output_folder
def wsi_to_png(input_folders, output_folder):
    # Get the base filename without the ".ndpi" extension
    for input_path in input_folders:
        base_filename = os.path.basename(input_path)
        #print(base_filename)
        try:
            # Open the WSI
            slide = openslide.OpenSlide(input_path)

            # Read a specific region from the image (entire slide) at level 6 dimension
            lvl = min(6,slide.level_count-1)
            #print(lvl)
            image = slide.get_thumbnail(size = slide.level_dimensions[lvl])

            # Define the output PNG file path with the modified filename
            output_path = os.path.join(output_folder, base_filename+'.png')
            image.save(output_path)
            # Save the image as PNG using cv2.imwrite
            
        except OpenSlideError as e:
            # This catches any OpenSlide-related errors, including unsupported format or corrupt files
            print(f"Failed to open {base_filename}: {e}")
        except UnidentifiedImageError as e:
            # This catches errors related to an unrecognized image format by PIL/Pillow
            print(f"Failed to open {base_filename} as an image file: {e}")



###############################################################################
### LOAD DATA
###############################################################################

# Define the output path and create temp folder for pngs
output_path = "/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/tissue_masks/100cohort/batch2"
#input_path = "/SAN/colcc/WSI_LymphNodes_BreastCancer/Mengyuan/smuLymphNet/sample_selection/sample_annotation/softlinks/selected"
#input_path = "/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/wsis/100cohort/batch2"
input_path = "/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/wsi_pngs/batch2"
model_path = "/SAN/colcc/WSI_LymphNodes_BreastCancer/models/LNsegmentation/20240227.model.keras"
#model_path = "/content/drive/MyDrive/LNseg/model/ananya.keras":

#print(model_path)

WSI = False


###### CREATE PNGS #############################
if WSI:
    temp_dir = os.path.join(output_path, "temp_pngs")
    os.makedirs(temp_dir, exist_ok=True)


    #read in wsi image paths
    wsi_paths = glob.glob(os.path.join(input_path, '*.ndpi'))
    wsi_paths = wsi_paths + glob.glob(os.path.join(input_path, '*.svs'))
    wsi_paths = wsi_paths + glob.glob(os.path.join(input_path, '*.mrxs'))
    #print(wsi_paths)

    # Read WSIs and write pngs to temp_dir
    wsi_to_png(wsi_paths, temp_dir)
    image_paths = glob.glob(os.path.join(temp_dir, '*.png')) 
else:
    #getting pngs directly
    image_paths = glob.glob(os.path.join(input_path, '*.png'))

print(image_paths)


###### PROCESS PNGS #############################

# Load data
wsi_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
wsi_dataset = wsi_dataset.map(lambda image_path: load_image(image_path))

# Preprocessing: Resize & Normalize
wsi_dataset = wsi_dataset.map(lambda x: resize(x["file_name"],x["image"],x["dims"]))
wsi_dataset = wsi_dataset.map(lambda x: normalize(x["file_name"],x["image"],x["dims"]))

# Prepare  batches
wsi_batches = wsi_dataset.batch(1)

print("done loading")

###############################################################################
### INFERENCE
###############################################################################

THRESH = 0.4

# Load trained model
loaded_model = tf.keras.models.load_model(model_path, custom_objects={'DiceLoss': DiceLoss},compile = False)
print("*** loaded model")
# Loop through each image
for img_obj in wsi_batches:
    #print(img_obj)
    #decode image file attributes
    image = img_obj['image']
    dims = img_obj['dims'][0].numpy()
    fname = img_obj['file_name']
    #print("*** i'm here 210: ",fname)
    fname = fname[0].numpy().decode('utf-8')
    #fname = os.path.basename(fname).replace('.png','')
    print(fname)
    print("*** about to call predict")
    # Predict the masks using the loaded model
    probabilities = loaded_model.predict(image)
    #print(np.sum(probabilities))
    #print(np.max(probabilities))
    print("*** finished predict")
    # Get binary predictions
    
    binary_masks = create_binary_mask(probabilities,THRESH)

    #need to resize the mask back to the original size before saving
    mask = binary_masks[0,:,:].numpy()
    mask = tf.image.resize(mask, (dims[0],dims[1]))

    #save mask to file
    cv2.imwrite(os.path.join(output_path,fname+'_lnmask.png'),mask.numpy())
    cv2.imwrite(os.path.join(output_path,fname+'_lnmask_viz.png'),mask.numpy()*255)

###################################################################################
### cleanup
###################################################################################
# Delete temp png dir
if WSI:
    shutil.rmtree(temp_dir)


