#!/bin/bash
#SBATCH --account=dsci410_510   
#SBATCH --partition=gpu     
#SBATCH --job-name=DactUnit_Tune    
#SBATCH --output=S_Bash/dact_bert_tune.out   
#SBATCH --error=S_Bash/dact_bert_tune.err    
#SBATCH --time=480                
#SBATCH --mem=32g              
#SBATCH --nodes=1               
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=4

#activate conda environment
module load miniconda3/20240410
conda activate moe_env

#run file
python Models/DactBert/dact_bert.py run