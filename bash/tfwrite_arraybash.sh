#!/bin/bash

#$ -l tmem=10G
#$ -l h_vmem=10G
#$ -l h_rt 02:50:00
#$ -j y
#$ -N 'arrayJobs' 

for f in 'g'; do
    for x in '10x'; do
        for s in '1536'; do
            echo $f
            echo $s
            echo $x 
            qsub /home/verghese/scripts/bash/tfwrite_bash.sh $f $x $s 
        done
    done
done
 
