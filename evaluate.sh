#!/bin/bash

#SBATCH --ntasks=1           ### How many CPU cores do you need?
#SBATCH --mem=14G            ### How much RAM memory do you need?
#SBATCH -p short             ### The queue to submit to: express, short, long, interactive
#SBATCH --gres=gpu:1         ### How many GPUs do you need?
#SBATCH -t 1-22:00:00        ### The time limit in D-hh:mm:ss format
#SBATCH -o /trinity/home/r103868/logs/out_%j.log
#SBATCH -e /trinity/home/r103868/logs/error_%j.log

module purge
module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

source /trinity/home/r103868/tomvenv/bin/activate

cd /trinity/home/r103868/bachelor-thesis/

# Check if the argument is provided
if [ -z "$1" ]; then
  echo "Usage: sbatch script.sh N"
  exit 1
fi

N=$1

# Loop from 1 to N
for i in $(seq 1 $N)
do
  echo "Running with seed $i"
   # python evaluate_bionet.py --seed $i
   python evaluate_logreg.py --seed $i
done