#!/bin/bash
#SBATCH --job-name=sink-softmax-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/qinch/code/sink/nanoGPT/logs/%x.out.%j
#SBATCH --error=/home/qinch/code/sink/nanoGPT/logs/%x.err.%j


module --ignore_cache load cuda/12.4
module --ignore_cache load cuda/"12.4"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch26-cuda124

cd /home/qinch/code/sink/nanoGPT
python compare.py --dataset data/wikitext/wikitext2.txt --compare --epochs 20
