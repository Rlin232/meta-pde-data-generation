#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH --time=4:00:00   # walltime
#SBATCH --mem=64G   # memory per CPU core
#SBATCH --gres gpu:1
#SBATCH -J "FENICS_SOLVER"   # job name
#SBATCH --mail-user=rylin@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

for i in {1..5000}
do
    # Original
    # python -m generate_data_2 --ground_truth_resolution 32 --siren_omega 30.0 --siren_omega0 30.0 --viz_every 10 --log_every 5 --measure_grad_norm_every 5 --grad_clip 100. --num_layers 3 --layer_size 64 --bc_weight 1.0 --outer_steps 200 --outer_points 1024 --validation_points 1024 --seed $i 

    python -m generate_data --ground_truth_resolution 32 --siren_omega 30.0 --siren_omega0 30.0 --viz_every 10 --log_every 5 --measure_grad_norm_every 5 --grad_clip 100. --num_layers 3 --layer_size 64 --bc_weight 1.0 --outer_steps 200 --outer_points 10000 --validation_points 2000 --seed $i 
done

# python -m generate_data --ground_truth_resolution 32 --siren_omega 30.0 --siren_omega0 30.0 --viz_every 10 --log_every 5 --measure_grad_norm_every 5 --grad_clip 100. --num_layers 3 --layer_size 64 --bc_weight 1.0 --outer_steps 200 --outer_points 10000 --validation_points 2000 --seed 0 
