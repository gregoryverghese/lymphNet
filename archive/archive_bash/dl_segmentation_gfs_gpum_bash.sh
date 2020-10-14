
#$ -l tmem=30G
#$ -l gpu=true
#$ -R y
#$ -l h_rt=20:00:00
#$ -j y
#$ -N 'dl_segment_unet_40' 

source /share/apps/source_files/python/python-3.7.0.source
source /share/apps/source_files/cuda/cuda-10.0.source
export LD_LIBRARY_PATH=/share/apps/TensorRT-5.1.5.0/lib:$LD_LIBRARY_PATH
python3 /home/verghese/scripts/python/ln_segmentation_gfs_test2.py -bp '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation' -ip 'three/images' -mp 'three/masks' -m 'unet' -n 'unet_40_gfs' -wf 'weight.csv' -pf 'params.json'

