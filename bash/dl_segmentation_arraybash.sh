#!/bin/bash

#$ -l tmem=5G
#$ -l h_vmem=5G
#$ -l h_rt 00:50:00
#$ -j y
#$ -N 'arrayJobs' 

for f in "g"; do
    for  ((i=1; i<=1; i++)); do
        qsub /home/verghese/scripts/bash/dl_segmentation.sh $f $i 
    done
done
 
