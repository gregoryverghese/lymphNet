#!/bin/bash

#$ -l tmem=10G
#$ -l h_vmem=10G
#$ -l h_rt 02:50:00
#$ -j y
#$ -N 'arrayJobs' 

for f in 'g'; do
    for x in '10x'; do
        for s in '1024'; do
            for o in '256'; do
	        echo $f
                echo $s
                echo $x
		echo $o
                qsub /home/verghese/breastcancer_ln_deeplearning/scripts/bash/tfwrite_bash.sh $f $x $s $o
            done
        done
    done
done
 
