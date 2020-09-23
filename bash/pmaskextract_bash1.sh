#$ -l tmem=30G
#$ -l h_vmem=30G
#$ -l h_rt 24:00:00
#$ -j y
#$ -N 'patchExtraction_5x_test' 

source /share/apps/source_files/python/python-3.7.0.source
export LD_LIBRARY_PATH=/share/apps/openslide-3.4.1/lib:$LD_LIBRARY_PATH
python3 ~/scripts/python/patchmask_extraction.py -np /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/ndpi/images -xp /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/wsi/Guys/sumans/xml -op /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/5x/one -cp /home/verghese/config/config_pextract_test.json -id images_g -md masks_g
