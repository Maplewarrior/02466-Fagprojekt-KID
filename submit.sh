#!/bin/sh
#BSUB -J aa                      # job name
#BSUB -o aa_%J.out               # output name
#BSUB -e aa_%J.err		 # output name (error)
#BSUB -q hpc                     # specify queue
#BSUB -W 24:00                       # set walltime limit hh:mm
#BSUB -R "rusage[mem=17GB]"     # memory per core/slot
#BSUB -R "span[hosts=1]"         # cores must be on the same host
#BSUB -n 29                       # number of cores
#BSUB -u s204122@dtu.dk          # email
# #BSUB -B			 # notify when start
#BSUB -N                       # notify when end


source env/bin/activate
cd 02466-Fagprojekt-KID/AAM-Module-V1
python3 result_script.py
