#!/bin/bash
#SBATCH -A saraansh
#SBATCH -n 5
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=48:00:00
#SBATCH --mincpus=5
#SBATCH --nodelist=gnode57

module add cuda/8.0
module add cudnn/7-cuda-8.0

python main.py \
							--train_dir /scratch/saraansh/Attribute-and-Simile-Classifiers-for-Face-Verification/train$3 \
							--test_dir /scratch/saraansh/Attribute-and-Simile-Classifiers-for-Face-Verification/test$3 \
							--save_path  /scratch/$1_neural_net_att_$2 \
							--att_file /scratch/saraansh/Attribute-and-Simile-Classifiers-for-Face-Verification/attr.pkl \
							--model $1 \
							--trait $2 | tee $1_neural_net_att_$2.txt
