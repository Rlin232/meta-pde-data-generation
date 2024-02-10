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
    FLAGS.n_eval = 1
    FLAGS.fixed_num_pdes = 1

    FLAGS(sys.argv) 
    
    key, subkey = jax.random.split(jax.random.PRNGKey(0))
    key, gt_key, gt_points_key = jax.random.split(key, 3)

    gt_keys = jax.random.split(gt_key, FLAGS.n_eval)
    gt_params = jax.vmap(sample_params)(gt_keys)

    pde = get_pde('poisson')

    params_list = jax_tools.tree_unstack(gt_params)

    fenics_functions, fenics_vals, coords, fenics_boundaries = trainer_util.get_ground_truth_points(
        pde, params_list, gt_points_key
    )

    N = len(fenics_functions)
    data = []
    keys = jax.random.split(gt_points_key, len(params_list))

    for i in range(min([N, 8])):
        # Extract keys, ground truth, and parameters for PDE
        key_i = keys[i]
        ground_truth = fenics_functions[i]
        boundary_mesh = fenics_boundaries[i]
        params = params_list[i]
        source_params, bc_params, geo_params = params

        # Extract coefficients for dataset reconstruction later
        c1, c2 = geo_params
        r0 = 1.0
        beta_i, gamma_i = source_params
        print(source_params)
        b_i = bc_params

        # Obtain datapoints and validation points
        train_points_boundary, train_points_domain = pde.sample_points(key_i, FLAGS.outer_points, params)
        ground_truth.set_allow_extrapolation(True)
        train_values_boundary = np.array([ground_truth(x) for x in train_points_boundary])[:, None]
        train_values_domain = np.array([ground_truth(x) for x in train_points_domain])[:, None]

        train_source_terms_domain = vmap_source(train_points_domain, source_params)
        train_source_terms_boundary = vmap_source(train_points_boundary, source_params)

        train_distances_boundary = np.array([0 for _ in train_points_boundary])[:, None]
        train_distances_domain = []
        for point in train_points_domain:
            x = point[0]
            y = point[1]

            distances = np.sqrt(np.square(boundary_mesh.coordinates()[:,0] - x) + np.square(boundary_mesh.coordinates()[:,1] - y))
            train_distances_domain.append(np.min(distances))
        train_distances_domain = np.array(train_distances_domain)[:,None]

        ground_truth.set_allow_extrapolation(False)


        val_points = coords[i]
        # Fenics_vals comes in the form [[val1], [val2], [val3], ...]
        val_values = fenics_vals[i].flatten()
        val_distances = []
        for point in val_points:
            x = point[0]
            y = point[1]

            distances = np.sqrt(np.square(boundary_mesh.coordinates()[:,0] - x) + np.square(boundary_mesh.coordinates()[:,1] - y))
            val_distances.append(np.min(distances))
        val_source_terms = vmap_source(val_points, source_params)

        point = train_points_domain[0]
        x = point[0]
        y = point[1]
        distances = np.sqrt(np.square(boundary_mesh.coordinates()[:,0] - x) + np.square(boundary_mesh.coordinates()[:,1] - y))

        # Plot and save ground truth visualization
        pde.plot_solution(ground_truth, params_list[i])

        # Code to visually check distance calculation 
        # plt.plot(x,y,'ro') 

        # ax = plt.gca()

        # circle = Circle(point, np.min(distances), edgecolor='r', facecolor='none')
        # ax.add_patch(circle)

        plt.title("Truth", fontsize=1)
        plt.savefig("viz_seed_{}.png".format(FLAGS.seed), dpi=800)

        # print(c1)
        # print(c2)
        # print(r0)
        # print(beta_i)
        # print(gamma_i)
        # print(b_i)

        coefs = {"seed": FLAGS.seed, "c1": c1, "c2": c2, "r0": r0, "beta": beta_i, "gamma": gamma_i, "b": b_i}

        data.append({"train_points_boundary": train_points_boundary, 
                     "train_values_boundary": train_values_boundary, 
                     "train_distances_boundary": train_distances_boundary,
                     "train_source_terms_boundary": train_source_terms_boundary,
                     "train_points_domain": train_points_domain, 
                     "train_values_domain": train_values_domain, 
                     "train_distances_domain": train_distances_domain,
                     "train_source_terms_domain": train_source_terms_domain,
                     "val_points": val_points, 
                     "val_values": val_values, 
                     "val_distances": val_distances,
                     "val_source_terms": val_source_terms,
                     "coefs": coefs
                     })
    

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
