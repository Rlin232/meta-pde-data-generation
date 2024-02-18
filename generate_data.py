import sys
from src.util import common_flags, trainer_util
from src.util import jax_tools

from src.get_pde import get_pde
from src.util.timer import Timer


import numpy as np
import jax
import torch

import matplotlib.pyplot as plt
import pdb

import fenics as fa

import pdb
from matplotlib.patches import Circle

from collections import namedtuple

from src.poisson.poisson_common import sample_params, vmap_source
from src.poisson.poisson_fenics import solve_fenics

from absl import app
from absl import flags

FLAGS = flags.FLAGS

if __name__ == "__main__":
    FLAGS.fixed_num_pdes = 1
    FLAGS(sys.argv) 
    data = []
    for i in range(FLAGS.n_eval):        
        key, subkey = jax.random.split(jax.random.PRNGKey(0))
        key, gt_key, gt_points_key = jax.random.split(key, 3)

        gt_keys = jax.random.split(gt_key, 1)
        gt_params = jax.vmap(sample_params)(gt_keys, np.repeat(i+1, gt_keys.shape[0]))

        pde = get_pde('poisson')

        params_list = jax_tools.tree_unstack(gt_params)

        fenics_functions, fenics_vals, coords, fenics_boundaries = trainer_util.get_ground_truth_points(
            pde, params_list, gt_points_key
        )

        N = len(fenics_functions)
        keys = jax.random.split(gt_points_key, len(params_list))

        # Extract keys, ground truth, and parameters for PDE
        key_i = keys[0]
        ground_truth = fenics_functions[0]
        boundary_mesh = fenics_boundaries[0]
        params = params_list[0]
        source_params, bc_params, geo_params = params

        # Extract coefficients for dataset reconstruction later
        c1, c2 = geo_params
        r0 = 1.0
        beta_i, gamma_i = source_params
        # print(source_params)
        b_i = bc_params

        # Obtain datapoints and validation points
        train_points_boundary, train_points_domain = pde.sample_points(key_i, FLAGS.outer_points, params)
        ground_truth.set_allow_extrapolation(True)
        train_values_boundary = np.array([ground_truth(x) for x in train_points_boundary])
        train_values_domain = np.array([ground_truth(x) for x in train_points_domain])

        train_source_terms_domain = vmap_source(train_points_domain, source_params)
        train_source_terms_boundary = vmap_source(train_points_boundary, source_params)

        train_distances_boundary = np.array([0 for _ in train_points_boundary])
        train_bc_boundary = train_values_boundary

        train_distances_domain = []
        train_bc_domain = []

        for point in train_points_domain:
            x = point[0]
            y = point[1]

            distances = np.sqrt(np.square(boundary_mesh.coordinates()[:,0] - x) + np.square(boundary_mesh.coordinates()[:,1] - y))
            
            min_distance = np.min(distances)
            min_index = np.where(distances == min_distance)[0]
            
            closest_boundary_point = (boundary_mesh.coordinates()[min_index,0], boundary_mesh.coordinates()[min_index,1])

            train_distances_domain.append(min_distance)
            train_bc_domain.append(ground_truth(closest_boundary_point))
        train_bc_domain = np.array(train_bc_domain)[:,None]
        train_distances_domain = np.array(train_distances_domain)[:,None]

        val_points = coords[0]
        # Fenics_vals comes in the form [[val1], [val2], [val3], ...]
        val_values = fenics_vals[0].flatten()
        val_distances = []
        val_bc = []
        for point in val_points:
            x = point[0]
            y = point[1]

            distances = np.sqrt(np.square(boundary_mesh.coordinates()[:,0] - x) + np.square(boundary_mesh.coordinates()[:,1] - y))
            
            min_distance = np.min(distances)
            min_index = np.where(distances == min_distance)[0]
            
            closest_boundary_point = (boundary_mesh.coordinates()[min_index,0], boundary_mesh.coordinates()[min_index,1])

            val_distances.append(np.min(distances))
            val_bc.append(ground_truth(closest_boundary_point))
        val_distances = np.array(val_distances)[:,None]
        val_bc = np.array(val_bc)[:,None]
        val_source_terms = vmap_source(val_points, source_params)

        ground_truth.set_allow_extrapolation(False)

        if FLAGS.n_eval <= 10:
            # Plot and save ground truth visualization
            pde.plot_solution(ground_truth, params_list[0])

            # Code to visually check distance calculation 
            # plt.plot(x,y,'ro') 
            # ax = plt.gca()
            # circle = Circle(point, np.min(distances), edgecolor='r', facecolor='none')
            # ax.add_patch(circle)

            plt.title("Truth", fontsize=1)
            plt.savefig("viz_seed_{}.png".format(i+1), dpi=800)

        # print(c1)
        # print(c2)
        # print(r0)
        # print(beta_i)
        # print(gamma_i)
        # print(b_i)

        coefs = {"seed": i+1, "c1": c1, "c2": c2, "r0": r0, "beta": beta_i, "gamma": gamma_i, "b": b_i}

        data.append({"train_points_boundary": train_points_boundary, 
                    "train_values_boundary": train_values_boundary, 
                    "train_distances_boundary": train_distances_boundary,
                    "train_source_terms_boundary": train_source_terms_boundary,
                    "train_bc_boundary": train_bc_boundary,
                    "train_points_domain": train_points_domain, 
                    "train_values_domain": train_values_domain, 
                    "train_distances_domain": train_distances_domain,
                    "train_source_terms_domain": train_source_terms_domain,
                    "train_bc_domain": train_bc_domain,
                    "val_points": val_points, 
                    "val_values": val_values, 
                    "val_distances": val_distances,
                    "val_source_terms": val_source_terms,
                    "coefs": coefs
                    })
        plt.figure().clear()
    

    torch.save(data, 'nonlinear_poisson.pt')

    # with Timer() as t:
    #     test_fns, test_vals, test_coords = trainer_util.get_ground_truth_points(
    #         pde,
    #         params_list,
    #         gt_points_key,
    #         #resolution=res,
    #         #boundary_points=int(FLAGS.boundary_resolution_factor * res),
    #         resolution=32,
    #         boundary_points=512
    #     )

    # for i, u in enumerate(test_fns):
    #     pde.plot_solution(u)
    #     plt.savefig(f"eval_{FLAGS.seed}.png", dpi=800)
