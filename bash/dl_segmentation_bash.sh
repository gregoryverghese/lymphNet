
#$ -l tmem=20G
#$ -l gpu=true
#$ -R y
#$ -l h_rt=03:00:00
#$ -j y
#$ -N 'unet_models' 

feature="$1"
iteration="$2" 
echo "$feature"
echo "$iteration"
 
source /share/apps/source_files/python/python-3.7.0.source
source /share/apps/source_files/cuda/cuda-10.0.source
export LD_LIBRARY_PATH=/share/apps/TensorRT-5.1.5.0/lib:$LD_LIBRARY_PATH
python3 /home/verghese/scripts/python/ln_segmentation_tf.py -rp /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/5x/one/tfrecords_"$feature" -cf config/config_bi_"$feature"_"$iteration".json -cp checkpoints -op predictions
