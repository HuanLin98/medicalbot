#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################


# (Should be normal, unless assigned a different one) The partition you've been assigned
#SBATCH --partition=normal

# How many nodes you require? You should only require 1.
#SBATCH --nodes=1

# When should you receive an email? sbatch --help for more options 
#SBATCH --mail-type=BEGIN,END,FAIL

# Number of CPU you'll like
#SBATCH --cpus-per-task=8

# Memory
#SBATCH --mem 12GB

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

# Who should receive the email notifications
#SBATCH --mail-user=huanlin.tay.2019@scis.smu.edu.sg

# How long do you require the resources? Note : If the job exceeds this run time, the manager will kill the job.
#SBATCH --time=23:00:00

# Do you require GPUS? If not, you should remove the following line (max GPU 1)
#SBATCH --gres=gpu:3090:1

#What is the QOS assigned to you? Check with myinfo command
#SBATCH --qos=cs425qos

# Where should the log files go?
# You must provide an absolute path eg /common/home/module/username/
# If no paths are provided, the output file will be placed in your current working directory
#SBATCH --output=%u.%j.out # STDOUT

# Give the job a name
#SBATCH --job-name=answer_training

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# We're not loading any Python modules because Python is automatically loaded

# Create a virtual environment
python3 -m venv /common/scratch/CS425/CS425G7/myenv

# This command assumes that you've already created the environment previously
# We're using an absolute path here. You may use a relative path, as long as SRUN is execute in the same working directory
source /common/scratch/CS425/CS425G7/myenv/bin/activate


# If you require any packages, install it as usual before the srun job submission.
# pip3 install pandas
# pip3 install numpy
# pip3 install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# pip3 install transformers

# Submit your job to the cluster
srun --gres=gpu:3090:1 python answer_mbert_training.py