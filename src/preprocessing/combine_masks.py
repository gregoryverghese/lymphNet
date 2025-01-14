"""
combine_masks.py: 
Combine LN segmentation mask and histoqc mask.

This script reads histoqc masks and LN segmentation masks, processes them to ensure they are binary, scales the LN segmentation masks, and combines them using a bitwise AND operation. The combined masks are then saved to an output directory.

Author:
    Holly Rafique

Contact:
    holly.rafique@kcl.ac.uk
"""

import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

import glob

from typing import List

HISTOQC_PATH: str = "D:\\03 Cancer Bioinformatics\\tissue_masks\\100cohort\\histoqc"
LN_SEG_PATH: str = "D:\\03 Cancer Bioinformatics\\tissue_masks\\100cohort\\using"
OUT_PATH: str = "D:\\03 Cancer Bioinformatics\\tissue_masks\\100cohort\\combined_with_hqc"
IS_BINARY: bool = True

def get_histoqc_paths(directory: str) -> List[str]:
    """
    Retrieve all file paths matching the histoqc mask pattern in the specified directory.

    Args:
        directory (str): Path to the directory containing histoqc masks.

    Returns:
        List[str]: List of file paths to histoqc masks.
    """
    return glob.glob(os.path.join(directory, '*.png'))

def process_and_combine_masks(histoqc_path: str, ln_seg_path: str, out_path: str, is_binary: bool) -> None:
    """
    Process and combine a histoqc mask and LN segmentation mask.

    Args:
        histoqc_path (str): Path to the histoqc mask file.
        ln_seg_path (str): Directory containing LN segmentation masks.
        out_path (str): Directory to save combined masks.
        is_binary (bool): Whether the LN segmentation mask is already binary.

    Returns:
        None
    """
    wsi_name = os.path.basename(histoqc_path).replace("_mask_use.png", "")
    print(f"Processing: {wsi_name}")

    # Read and process histoqc mask
    histoqc_mask = cv2.imread(histoqc_path, cv2.IMREAD_GRAYSCALE)
    _, histoqc_binary = cv2.threshold(histoqc_mask, 5, 255, cv2.THRESH_BINARY)

    # Read LN segmentation mask
    ln_mask_path = os.path.join(ln_seg_path, f"{wsi_name}.png_lnmask.png")
    ln_mask = cv2.imread(ln_mask_path, cv2.IMREAD_GRAYSCALE)

    if ln_mask is None:
        print(f"Warning: LN mask for {wsi_name} not found at {ln_mask_path}.")
        return

    if is_binary:
        ln_binary = ln_mask
    else:
        _, ln_binary = cv2.threshold(ln_mask, 127, 255, cv2.THRESH_BINARY)

    # Resize LN mask to match histoqc mask size
    ln_resized = cv2.resize(ln_binary, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

    # Combine the masks
    combined_mask = cv2.bitwise_and(histoqc_binary, ln_resized)

    # Save combined mask and visualization
    combined_mask_path = os.path.join(out_path, f"{wsi_name}_tissuemask.png")
    viz_mask_path = os.path.join(out_path, f"VIZ_{wsi_name}.png")

    cv2.imwrite(combined_mask_path, combined_mask)
    cv2.imwrite(viz_mask_path, combined_mask * 255)

def main():
    """
    Main function to process all histoqc masks and combine them with LN segmentation masks.

    Returns:
        None
    """
    hqc_paths = get_histoqc_paths(HISTOQC_PATH)

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for histoqc_path in hqc_paths:
        process_and_combine_masks(histoqc_path, LN_SEG_PATH, OUT_PATH, IS_BINARY)

if __name__ == "__main__":
    main()



