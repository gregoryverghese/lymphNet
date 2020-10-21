#added script to calculate std and mean of training set used to normalize images (values are stated in config file nunder normalize)
#added ability to standardize images on the fly in the tfrecord_read file (need to finish adding this to predict script
#refactored predict script slightly but still essentially running the same thing
#tidied up the patchmask_extraction.py  script a bit and changed name to patching.py (whole script needs refactoring though)
#moved old scripts into archive folder
#change the calculate_weight script to calculate weights slightly differently (before using scikit_learn class_weights function which led to large outliers)
#moved old calulcate weight script into archive folder
#some bash files may need to change to reflect changes in script names

#currently running through and working with the following 

python3 /home/verghese/breastcancer_ln_deeplearning/scripts/python/tuning.py -rp /SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/2.5x/one/data_3/tfrecords -rd tfrecords_g_256_32 -cf /home/verghese/breastcancer_ln_deeplearning/scripts/config/config_germinal_template.json -cp checkpoints -op /home/verghese/breastcancer_ln_deeplearning/output/ -mn unet

#Add your chanegs where neccessary  and then we can go over these scripts tomorrow with paul 
