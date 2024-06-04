"""
" wsi_to_png.ipynb
" author: Holly Rafique
" date: 22/02/2024
" 
" based on script by Dina
" converts Whole Slide Image to png
"
" input: ndpi files of H&E stained WSIs
" output: png version
"""

# Import necessary modules
import os
import openslide
from PIL import Image
import glob
import cv2
import numpy as np
from PIL import UnidentifiedImageError
from openslide import OpenSlideError

# Define a function to convert and save an NDPI image as PNG
def convert_ndpi_to_png(input_path, output_path):
    #get the base filename without the extension
    fname = os.path.basename(input_path)
    #fname = fname.replace('.ndpi', '')
    #fname = fname.replace('.svs', '')
    #fname = fname.replace('.mrxs', '')
    print(fname)
    try:
        # Open the NDPI image
        slide = openslide.OpenSlide(input_path)
  
        # Read a specific region from the image (entire slide) at level 6 dimension
        img_level = min(6,slide.level_count-1)
        print(slide.level_count)
        print(img_level)
        print(slide.level_dimensions[img_level])
        image = slide.get_thumbnail(size = slide.level_dimensions[img_level])
    
        # Convert image to numpy array
        image_np = np.array(image.convert('RGB'))

        # Save the image as PNG using cv2.imwrite
        cv2.imwrite(os.path.join(output_path, fname+".png"), image_np)
    except OpenSlideError as e:
        # This catches any OpenSlide-related errors, 
         #including unsupported format or corrupt files
        print(f"Failed to open {fname}: {e}")
    except UnidentifiedImageError as e:
        # This catches errors related to an unrecognized image format by PIL/Pillow
        print(f"Failed to open {fname} as an image file: {e}")

# Folder path with ndpi WSIs
input_path = "/SAN/colcc/WSI_LymphNodes_BreastCancer/Mengyuan/smuLymphNet/sample_selection/sample_annotation/softlinks/selected"

# Store all the image paths in a list
#ndpi_paths = glob.glob(os.path.join(input_path, '*.ndpi'))
svs_paths = glob.glob(os.path.join(input_path, '*.svs'))
#mrxs_paths = glob.glob(os.path.join(input_path, '*.mrxs'))
#image_paths = ndpi_paths + svs_paths+ mrxs_paths
image_paths = svs_paths

# Define the output folder
output_path = "/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/wsi_pngs/100cohort"

print("paths loaded")
# Loop through each image in the list
for i in image_paths: 
    #print(i)
    # Call the function to convert each NDPI image to PNG format
    convert_ndpi_to_png(i, output_path)

# Print message when all images are being 
print("done") 



