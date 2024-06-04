import os
import glob
import numpy as np
import cv2
import argparse

def stitch(input_path, output_path,patch_size=1024, step_size=512):
 

	# Get the list of patch files in the directory
	patch_files = glob.glob(os.path.join(input_path,'*.png'))
	print(patch_files)
	patches = {}

	for file_path in patch_files:
		file_name = os.path.basename(file_path)[:-4]  # Get the file name from the path
		imagename, xpos, ypos = file_name.split('_')
		if imagename not in patches:
			patches[imagename] = []
		patches[imagename].append([file_name,int(int(xpos)/4),int(int(ypos)/4)])

	imgsize={}
	print("\n\n")
	for k in patches:
		print(k)
		k_patches = patches[k]
		k_patches.sort(key=lambda x: x[1])
		xmin = min([p[1] for p in k_patches])
		xmax = max([p[1] for p in k_patches])+patch_size
		ymin = min([p[2] for p in k_patches])
		ymax = max([p[2] for p in k_patches])+patch_size
		w = int((xmax-xmin))
		h = int((ymax-ymin))
		imgsize[k] = [(w,h),xmin,ymin]
		print(imgsize[k])
		patches[k] = k_patches
		print(xmin, xmax, ymin, ymax)

	for k in patches:
		print(k)
		stitched_size = imgsize[k][0]
		# Create an empty array to store the stitched image
		stitched_image = np.ones((stitched_size[1], stitched_size[0], 3), dtype=np.uint8) *255
		print(stitched_image.shape)
		# Iterate through each patch and stitch them
		#    # Calculate the position of the patch in the stitched image
		#    row = (i // ((stitched_size - patch_size) // step_size + 1)) * step_size
		#    col = (i % ((stitched_size - patch_size) // step_size + 1)) * step_size
		xstart=imgsize[k][1]
		ystart=imgsize[k][2]

		for i,patch in enumerate(patches[k]):
			#patch is a list of [filename,xstart,ystart]
			patch_img = cv2.imread(os.path.join(input_path,patch[0]+".png"))
			# Calculate the position of the patch in the stitched image
			row = patch[1]-xstart
			col = patch[2]-ystart
			# Place the patch in the stitched image
			stitched_image[col:col+patch_size,row:row+patch_size] = patch_img

		#really want to add the datetime to the file
		cv2.imwrite(os.path.join(output_path,k+".png"), stitched_image)

	print("finished")



if __name__=='__main__':
    ap=argparse.ArgumentParser(description='stitch patches together')
    ap.add_argument('-ip','--input_path',required=True,help='path to patches to stitch')
    ap.add_argument('-op','--output_path',required=True,help='path to save stitched image')
    ap.add_argument('-ps', '--patch_size',default=1024)
    ap.add_argument('-ss', '--step_size',default=512)
    args=ap.parse_args()

    stitch(args.input_path, args.output_path)


#patch_directory = '/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/patches/100cohort/batch1/images'
#output_file='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/100-cohort/stitched'

#patch_size = 1024
#step_size = 512
