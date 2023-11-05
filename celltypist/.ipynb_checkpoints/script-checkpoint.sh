#!/bin/bash
#BSUB -J celltypist_t1
#BSUB -n 2
#BSUB -R span[ptile=6]
#BSUB -R rusage[mem=150]
#BSUB -W 8:00
#BSUB -o /data/peer/adamsj5/cell_typing/bsub_outputs/%J.stdout
#BSUB -e /data/peer/adamsj5/cell_typing/bsub_outputs/%J.stderr

cd $LS_SUBCWD
module load python cuda/11.3
conda activate base
python -u /home/adamsj5/auto_cell_typing/celltypist/train_models_celltypist.py