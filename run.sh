#!/bin/bash
#SBATCH -A saraansh
#SBATCH -n 1
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=2048
#SBATCH --time=48:00:00
#SBATCH --mincpus=1

python attribute_clf_new.py \
		--trait $1 | tee attr_clf_$1.txt