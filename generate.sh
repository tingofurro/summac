#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --time=1-00:00:00


#SBATCH --job-name=gen_la_full

module load Anaconda3/2022.10
module load CUDA/11.8.0
source activate seq


python generate_summary_fast.py --prune_method sparsegpt --data polytope \
                                --model_name "facebook/opt-iml-1.3b"


