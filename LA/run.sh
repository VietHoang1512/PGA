#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=180GB
#SBATCH --job-name=UDA
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate /scratch/envs/python
nvidia-smi

cd /scratch/implement/UDA/LA/

args=("$@")
echo $# arguments passed
# echo ${args[0]} ${args[1]} ${args[2]} 
# radius=${args[0]}
# tradeoff=${args[1]}
# align=${args[2]}

radius=0.0
align=.0001
tradeoff=100.


# python main.py  --entropy_tradeoff .0 --radius $radius --tradeoff $tradeoff --align $align --M1 16 --M2 16 --threshold .4 --data_root data/image-clef --dataset ImageCLEF

# for tradeoff in  1  
# do
# python main.py  --entropy_tradeoff .0 --radius $radius --tradeoff $tradeoff --align $align --M1 16 --M2 16 --threshold .4 --data_root /vast/data/image-clef --dataset ImageCLEF
# done

# python main.py  --entropy_tradeoff .0 --radius .0001 --tradeoff .1 --align .0 --M1 16 --M2 16 --threshold .4 --data_root data/image-clef --dataset ImageCLEF

# python main.py  --entropy_tradeoff .0 --radius $radius --tradeoff $tradeoff --align $align --M1 16 --M2 16 --threshold .4 --data_root data/office_home/ --dataset OfficeHome

# python main_uda.py  --entropy_tradeoff .0 --radius $radius --tradeoff $tradeoff --align $align --M1 16 --M2 16 --threshold .4 --data_root data/office_home/ --dataset OfficeHome
# python main_multi.py  --entropy_tradeoff .0 --radius $radius --tradeoff $tradeoff --align $align --M1 16 --M2 16 --threshold .4 --data_root data/office_home/ --dataset OfficeHome

# python main.py  --entropy_tradeoff .0 --radius $radius --tradeoff $tradeoff --align $align --M1 16 --M2 16 --threshold .4 --data_root /vast/data/domainnet --dataset DomainNet
python main_multi.py  --entropy_tradeoff .0 --radius $radius --tradeoff $tradeoff --align $align --M1 16 --M2 16 --threshold .4 --data_root /vast/data/domainnet --dataset DomainNet

# python main.py  --entropy_tradeoff .0 --radius $radius --tradeoff $tradeoff --align $align --M1 16 --M2 16 --threshold .4 --data_root data/PACS --dataset PACS

