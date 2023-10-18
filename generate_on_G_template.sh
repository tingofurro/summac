#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=150G
#SBATCH --time=1-00:00:00


#SBATCH --job-name=gen_la_full

module load Anaconda3/2022.10
module load CUDA/11.8.0
source activate seq

for method in "wanda" "fullmodel" "sparsegpt"
do
for prompt_id in 1 2 3 4 5
do
python generate_summary_fast.py --prune_method $method --data polytope \
                                --model "decapoda-research/llama-30b-hf" \
                                --prompt_id None 

done
done
