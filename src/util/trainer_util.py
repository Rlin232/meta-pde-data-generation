"""Util functions used by both Leap and Maml which dont have another home.

Consider moving these into subfolders.

Also try move more functions in here."""
import pickle

import jax
import jax.numpy as np
from jax import vmap
import numpy as npo
import matplotlib.pyplot as plt
import flax

import os
import shutil
from .tensorboard_logger import Logger as TFLogger

# import flaxOptimizers
from functools import partial

import fenics as fa
import dolfin

from . import jax_tools
import copy
import pdb

import absl
from absl import app
from absl import flags

FLAGS = flags.FLAGS


def get_ground_truth_points(
    pde, pde_params_list, key, resolution=None, boundary_points=None
):
    """Given a pdedef and list of pde parameters, sample points in the domain and
    evaluate the ground truth at those points."""
    fenics_functions = []
    fenics_boundaries = []
    coefs = []
    coords = []
    keys = jax.random.split(key, len(pde_params_list))

    if resolution is None:
        resolution = FLAGS.ground_truth_resolution

    if boundary_points is None:
        boundary_points = int(
            FLAGS.boundary_resolution_factor * FLAGS.ground_truth_resolution
        )

    for params, key in zip(pde_params_list, keys):
        ground_truth, bbtree = pde.solve_fenics(
            params, resolution=resolution, boundary_points=boundary_points
        )
        k1, k2 = jax.random.split(key)
        fn_coords = pde.sample_points_in_domain(k1, FLAGS.validation_points, params)
        #fn_coords = np.concatenate(pde.sample_points(k1, FLAGS.validation_points, params))
        if FLAGS.pde == 'td_burgers':
            # replace random time sampling with fenics val sampled time
            tile_idx = (FLAGS.validation_points // FLAGS.num_tsteps) + 1
            time_axis = np.tile(
                np.array(ground_truth.timesteps_list),
                tile_idx,
            )[0: fn_coords.shape[0]]
            fn_coords = np.concatenate([fn_coords[:, :-1], time_axis[:, None]], axis=1)

        ground_truth.set_allow_extrapolation(True)
        coefs.append(np.array([ground_truth(x) for x in fn_coords])[:, None])
        coords.append(fn_coords)
        ground_truth.set_allow_extrapolation(False)
        fenics_functions.append(ground_truth)
        fenics_boundaries.append(bbtree)
    return fenics_functions, np.stack(coefs, axis=0), np.stack(coords, axis=0), fenics_boundaries


def save_fenics_solution(cache, fenics_function):
    """
    Args:
        cache: dictionary
        fenics_function: fenics solution or ground truths
    """
    pde = cache['pde']
    out_dir = f"{pde}_fenics_solutions"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    suffix = len(os.listdir(out_dir))
    path = os.path.join(out_dir, str(suffix))
    assert not os.path.exists(path)
    if not os.path.exists(os.path.join(out_dir, "master_info.pickle")):
        master_info = {
            str(suffix): (cache['hparams'], cache["params"])
        }
        with open(os.path.join(out_dir, "master_info.pickle"), 'wb') as handle:
            pickle.dump(master_info, handle)
    else:
        with open(os.path.join(out_dir, "master_info.pickle"), 'rb') as handle:
            master_info = pickle.load(handle)
        master_info[str(suffix)] = \
            (cache['hparams'], cache["params"])

        with open(os.path.join(out_dir, "master_info.pickle"), 'wb') as handle:
            pickle.dump(master_info, handle)

    os.mkdir(path)

    if type(fenics_function) == dolfin.function.function.Function:
        assert pde != 'td_burgers'
        with dolfin.cpp.io.XDMFFile(os.path.join(path, f'pde')) as f:
            f.write_checkpoint(fenics_function, f"{cache['pde']}")
    else:
        assert pde == 'td_burgers'
        for i, fenics_function_t in enumerate(fenics_function):
            with dolfin.cpp.io.XDMFFile(os.path.join(path, f'pde_{i}')) as f:
                f.write_checkpoint(fenics_function_t, f"{pde}_{i}")

    return path


def read_fenics_solution(cache, fenics_function):
    """
    Args:
        cache: dictionary
        fenics_function: place holder for the fenics solution
    """
    pde = cache['pde']
    out_dir = f"{pde}_fenics_solutions"
    if not os.path.exists(out_dir):
        return False
    else:
        with open(os.path.join(out_dir, "master_info.pickle"), 'rb') as handle:
            master_info = pickle.load(handle)
        suffix = None
        for suffix_i, cache_i in master_info.items():
            if (
                    npo.isclose(cache_i[0], cache['hparams']).all()
                    and npo.array([npo.allclose(a, b) for a, b in zip(cache_i[1], cache['params'])]).all()
            ):
                suffix = suffix_i
                break

        if (suffix is not None) and (type(fenics_function) == list):
            path = os.path.join(out_dir, suffix)
            for i in range(len(fenics_function)):
                with dolfin.cpp.io.XDMFFile(os.path.join(path, f'pde_{i}')) as handle:
                    handle.read_checkpoint(fenics_function[i], f"{cache['pde']}_{i}")
            return True

        elif (suffix is not None) and (type(fenics_function) == dolfin.function.function.Function):
            path = os.path.join(out_dir, suffix)
            with dolfin.cpp.io.XDMFFile(os.path.join(path, f'pde')) as handle:
                handle.read_checkpoint(fenics_function, f"{cache['pde']}")
            return True

        else:
            return False


def extract_coefs_by_dim(function_space, dofs, out, dim_idx=0):
    """Given the value of u (in R^d) at *each* dof coordinate, fill into 'out' the
    vector of dof values corresponding to interpolating u into function_space.

    We need this funtion because each dof corresponds to only one dim of u(x),
    so evaluating u(x) at every dof coordinate gives us d times as many values as
    needed.

    Depending on the function space, dof coordinates might not be repeated (e.g.
    at some x, there is only a dof for some but not all dimensions), and we need to
    use fenics API to extract the right components of u for that dof).
    """
    dofs = dofs.reshape(dofs.shape[0], -1)  # If DoFs is 1d, make it 2d
    if function_space.num_sub_spaces() > 0:
        for sidx in range(function_space.num_sub_spaces()):
            dim_idx = extract_coefs_by_dim(function_space.sub(sidx), dofs, out, dim_idx)
    else:
        if dim_idx >= dofs.shape[1]:
            raise Exception("Mismatch between Fenics fn space dim and dofs arr")
        out[function_space.dofmap().dofs()] = dofs[
            function_space.dofmap().dofs(), dim_idx
        ]
        dim_idx += 1
    return dim_idx


def compare_plots_with_ground_truth(
    model,
    pde,
    fenics_functions,
    params_stacked,
    get_final_model,
    meta_alg_def,
    inner_steps,
):
    """Plot the solutions corresponding to ground truth, and corresponding to the
    meta-learned model after each of k gradient steps.
    """
    keys = jax.random.split(jax.random.PRNGKey(0), len(fenics_functions))
    N = len(fenics_functions)

    params_list = jax_tools.tree_unstack(params_stacked)

    for i in range(min([N, 8])):  # Don't plot more than 8 PDEs for legibility
        ground_truth = fenics_functions[i]
        plt_inner_steps = min(max(inner_steps, 1), 5)
        plt.subplot(plt_inner_steps + 1, min([N, 8]), 1 + i)
        if N > 1:
            plt.axis("off")
        plt.xlim([min(FLAGS.xmin * 0.9, FLAGS.xmin - (FLAGS.xmax - FLAGS.xmin) * 0.1),
                  FLAGS.xmax * 1.1])
        plt.ylim([min(FLAGS.ymin * 0.9, FLAGS.ymin - (FLAGS.ymax - FLAGS.ymin) * 0.1),
                  FLAGS.ymax * 1.1])
        ground_truth.set_allow_extrapolation(False)
        pde.plot_solution(ground_truth, params_list[i])
        if i == 0:
            plt.title("Truth", fontsize=4)
        steps_plot = np.linspace(0, inner_steps, plt_inner_steps, dtype=int)
        for j, step in enumerate(steps_plot): #range(0, inner_steps + 1):
            plt.subplot(plt_inner_steps + 1, min([N, 8]), 1 + min([N, 8]) * (j + 1) + i)
            plt.xlim([min(FLAGS.xmin * 0.9, FLAGS.xmin - (FLAGS.xmax - FLAGS.xmin) * 0.1),
                      FLAGS.xmax * 1.1])
            plt.ylim([min(FLAGS.ymin * 0.9, FLAGS.ymin - (FLAGS.ymax - FLAGS.ymin) * 0.1),
                      FLAGS.ymax * 1.1])

            final_model = get_final_model(
                keys[i], model, params_list[i], step, meta_alg_def,
            )

            coords = ground_truth.function_space().tabulate_dof_coordinates()

            # supplement time-axis for td_burgers
            if FLAGS.pde == 'td_burgers':
                assert coords.shape[1] == 2
                t_list = np.linspace(FLAGS.tmin, FLAGS.tmax, FLAGS.num_tsteps, endpoint=False)
                t_idx = npo.unique(npo.linspace(0, len(t_list), 2, endpoint=False, dtype=int)[1:])
                t_supp = np.squeeze(t_list[t_idx])
                print(f'plotting final model at t = {t_supp:.2f}')
                t_supp = np.repeat(t_supp, coords.shape[0]).reshape(coords.shape[0], 1)

                coords_t = np.concatenate([coords, t_supp], axis=1)

                dofs = final_model(np.array(coords_t))
            elif FLAGS.pde == 'hyper_elasticity':
                ground_truth_vals = np.array([ground_truth(x) for x in coords])
                dofs_left = final_model(np.array(coords))
                mse_left = np.mean((dofs_left - ground_truth_vals) ** 2)
                
                coords_right = np.array(coords)
                coords_right = coords_right.at[:, 0].set(1. - coords_right.at[:, 0].get())
                dofs_right = final_model(np.array(coords_right))
                dofs_right = dofs_right.at[:, 0].multiply(-1.)
                mse_right = np.mean((dofs_right - ground_truth_vals) ** 2)

                dofs = jax.lax.cond(
                    mse_left > mse_right,
                    lambda _: dofs_right,
                    lambda _: dofs_left,
                    None
                )
            else:
                dofs = final_model(np.array(coords))

            out = npo.zeros(len(coords))

            extract_coefs_by_dim(ground_truth.function_space(), dofs, out)

            u_approx = fa.Function(ground_truth.function_space())

            fenics_shape = npo.array(u_approx.vector()).shape
            assert fenics_shape == out.shape, "Mismatched shapes {} and {}".format(
                fenics_shape, out.shape
            )

            u_approx.vector().set_local(out)

            pde.plot_solution(u_approx, params_list[i])
            if (i == 0) and (j == 0):
                plt.title("NN Model", fontsize=4)
                if N > 1:
                    plt.tick_params(axis='both', length=0, labelsize=1, colors='white')
                    plt.ylabel(f"Step {step}", fontsize=4)
                else:
                    plt.tick_params(axis='both', length=0, labelsize=5, colors='black')
            elif i == 0:
                if N > 1:
                    plt.tick_params(axis='both', length=0, labelsize=1, colors='white')
                    plt.ylabel(f"Step {step}", fontsize=4)
                else:
                    plt.tick_params(axis='both', length=0, labelsize=5, colors='black')
            else:
                plt.axis("off")


def plot_model_time_series(
    model,
    pde,
    fenics_functions,
    params_stacked,
    get_final_model,
    meta_alg_def,
    inner_steps,
):
    """plot time series gif
    """
    keys = jax.random.split(jax.random.PRNGKey(0), len(fenics_functions))
    N = len(fenics_functions)

    params_list = jax_tools.tree_unstack(params_stacked)

    tmp_filenames = []
    for t in range(FLAGS.num_tsteps):
        plt.figure()
        for i in range(min([N, 8])):  # Don't plot more than 8 PDEs for legibility
            ground_truth = fenics_functions[i]
            plt_inner_steps = min(max(inner_steps, 1), 5)
            plt.subplot(plt_inner_steps + 1, min([N, 8]), 1 + i)
            if N > 1:
                plt.axis("off")
            plt.xlim([min(FLAGS.xmin * 0.9, FLAGS.xmin - (FLAGS.xmax - FLAGS.xmin) * 0.1),
                      FLAGS.xmax * 1.1])
            #plt.ylim([min(FLAGS.ymin * 0.9, FLAGS.ymin - (FLAGS.ymax - FLAGS.ymin) * 0.1),
            #          FLAGS.ymax * 1.1])

            ground_truth.set_allow_extrapolation(False)
            t_val = ground_truth.timesteps_list[t]
            pde.plot_solution(ground_truth[t], params_list[i])
            if i == 0:
                plt.title("t = {:.2f} \n Truth".format(t_val), fontsize=6)
            #for j in range(0, inner_steps + 1):
            steps_plot = np.linspace(0, inner_steps, plt_inner_steps, dtype=int)
            for j, step in enumerate(steps_plot):
                plt.subplot(plt_inner_steps + 1, min([N, 8]), 1 + min([N, 8]) * (j + 1) + i)
                plt.xlim([min(FLAGS.xmin * 0.9, FLAGS.xmin - (FLAGS.xmax - FLAGS.xmin) * 0.1),
                          FLAGS.xmax * 1.1])
                #plt.ylim([min(FLAGS.ymin * 0.9, FLAGS.ymin - (FLAGS.ymax - FLAGS.ymin) * 0.1),
                #          FLAGS.ymax * 1.1])

                final_model = get_final_model(
                    keys[i], model, params_list[i], j, meta_alg_def,
                )

                coords = ground_truth.function_space().tabulate_dof_coordinates()

                # add time dimension
                t_val_expand = np.repeat(t_val, len(coords)).reshape(-1, 1)
                coords_t = np.concatenate([coords, t_val_expand], axis=1)

                dofs = final_model(np.array(coords_t))

                out = npo.zeros(len(coords))

                extract_coefs_by_dim(ground_truth.function_space(), dofs, out)

                u_approx = fa.Function(ground_truth.function_space())

                fenics_shape = npo.array(u_approx.vector()).shape
                assert fenics_shape == out.shape, "Mismatched shapes {} and {}".format(
                    fenics_shape, out.shape
                )

                u_approx.vector().set_local(out)

                pde.plot_solution(u_approx, params_list[i])

                if (i == 0) and (j == 0):
                    plt.title("NN Model", fontsize=4)
                    if N > 1:
                        plt.tick_params(axis='both', length=0, labelsize=1, colors='white')
                        plt.ylabel(f"Step {step}", fontsize=4)
                    else:
                        plt.tick_params(axis='both', length=0, labelsize=5, colors='black')
                elif i == 0:
                    if N > 1:
                        plt.tick_params(axis='both', length=0, labelsize=1, colors='white')
                        plt.ylabel(f"Step {step}", fontsize=4)
                    else:
                        plt.tick_params(axis='both', length=0, labelsize=5, colors='black')
                else:
                    plt.axis("off")
        # save fig here
        plt.savefig('timedependent_burger_' + str(t) + '.png', dpi=400)
        plt.close()
        tmp_filenames.append('timedependent_burger_' + str(t) + '.png')
        #build_gif(tmp_filenames)
    return tmp_filenames


def plot_model_time_series_new(
    model,
    pde,
    fenics_functions,
    params_stacked,
    get_final_model,
    meta_alg_def,
    inner_steps,
):
    """plot time series 2d
    """
    keys = jax.random.split(jax.random.PRNGKey(0), len(fenics_functions))
    N = len(fenics_functions)

    params_list = jax_tools.tree_unstack(params_stacked)

    for i in range(min([N, 8])):  # Don't plot more than 8 PDEs for legibility
        ground_truth = fenics_functions[i]
        plt_inner_steps = min(max(inner_steps, 1), 5)
        plt.subplot(plt_inner_steps + 1, min([N, 8]), 1 + i)
        if N > 1:
            plt.axis("off")

        fenics_vals = []
        for t in range(FLAGS.num_tsteps):
            ground_truth.set_allow_extrapolation(True)
            x_coords = np.linspace(FLAGS.xmin, FLAGS.xmax, FLAGS.validation_points)

            fenics_vals.append(
                np.array([ground_truth[t](xi) for xi in x_coords])
            )
        fenics_vals = np.array(fenics_vals)
        clr = plt.imshow(fenics_vals.T, cmap='rainbow', interpolation='none', aspect='auto')
        plt.xticks(np.linspace(0, FLAGS.num_tsteps, 11), np.linspace(0, FLAGS.tmax, 11), fontsize=4)
        plt.yticks(np.linspace(0, FLAGS.validation_points, 11), np.linspace(FLAGS.xmin, FLAGS.xmax, 11), fontsize=4)
        plt.ylabel('position', fontsize=4)
        cbar = plt.colorbar(clr)
        cbar.ax.tick_params(labelsize=4)
        if i == 0:
            plt.title("Truth", fontsize=6)

        steps_plot = np.linspace(0, inner_steps, plt_inner_steps, dtype=int)
        for j, step in enumerate(steps_plot):
            model_vals = []
            plt.subplot(plt_inner_steps + 1, min([N, 8]), 1 + min([N, 8]) * (j + 1) + i)
            for t in range(FLAGS.num_tsteps):
                final_model = get_final_model(
                    keys[i], model, params_list[i], j, meta_alg_def,
                )
                # add time dimension
                t_val = ground_truth.timesteps_list[t]
                t_val_expand = np.repeat(t_val, len(x_coords)).reshape(-1, 1)
                coords = np.concatenate([x_coords[:, None], t_val_expand], axis=1)

                model_vals.append(final_model(np.array(coords)))

            model_vals = np.array(model_vals)
            clr = plt.imshow(model_vals.T, cmap='rainbow', interpolation='none', aspect='auto')
            plt.xticks(np.linspace(0, FLAGS.num_tsteps, 11), np.linspace(0, FLAGS.tmax, 11), fontsize=4)
            plt.xlabel('Time (s)', fontsize=4)
            plt.yticks(np.linspace(0, FLAGS.validation_points, 11), np.linspace(FLAGS.xmin, FLAGS.xmax, 11), fontsize=4)
            plt.ylabel('position', fontsize=4)
            cbar = plt.colorbar(clr)
            cbar.ax.tick_params(labelsize=4)

            if (i == 0) and (j == 0):
                plt.title("NN Model", fontsize=4)
                if N > 1:
                    plt.tick_params(axis='both', length=0, labelsize=1, colors='white')
                    plt.ylabel(f"Step {step}", fontsize=4)
                else:
                    plt.tick_params(axis='both', length=0, labelsize=5, colors='black')
            elif i == 0:
                if N > 1:
                    plt.tick_params(axis='both', length=0, labelsize=1, colors='white')
                    plt.ylabel(f"Step {step}", fontsize=4)
                else:
                    plt.tick_params(axis='both', length=0, labelsize=5, colors='black')
            else:
                plt.axis("off")

    return


def prepare_logging(out_dir, expt_name):
    if expt_name is not None:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        path = os.path.join(out_dir, expt_name)
        if os.path.exists(path):
            shutil.rmtree(path)
        if not os.path.exists(path):
            os.mkdir(path)

        outfile = open(os.path.join(path, "log.txt"), "w")

        def log(*args, **kwargs):
            print(*args, **kwargs, flush=True)
            print(*args, **kwargs, file=outfile, flush=True)

        tflogger = TFLogger(path)

    else:

        def log(*args, **kwargs):
            print(*args, **kwargs, flush=True)

        tflogger = None

    return path, log, tflogger


@partial(jax.jit, static_argnums=[4])
def vmap_validation_error(
    model, ground_truth_params, points, ground_truth_vals, make_coef_func
):
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, FLAGS.n_eval)

    coefs = vmap(make_coef_func, (0, None, 0, 0))(
        keys, model, ground_truth_params, points
    )

    coefs = coefs.reshape(coefs.shape[0], coefs.shape[1], -1)
    ground_truth_vals = ground_truth_vals.reshape(coefs.shape)
    # if linear stokes, center mean pressure to 0 before comparing
    if FLAGS.pde == 'linear_stokes':
        err_velocity = coefs[:, :, :-1] - ground_truth_vals[:, :, :-1]
        coefs_p = coefs[:, :, -1] - np.mean(coefs[:, :, -1])
        ground_truth_vals_p = ground_truth_vals[:, :, -1] - np.mean(ground_truth_vals[:, :, -1])
        err_pressure = (coefs_p - ground_truth_vals_p)[:, :, np.newaxis]
        err = np.concatenate([err_velocity, err_pressure], axis=2)
        mse = np.mean(err ** 2)
        normalizer = np.mean(ground_truth_vals ** 2, axis=1, keepdims=True)
        rel_sq_err = err ** 2 / normalizer.mean(axis=2, keepdims=True)

    elif FLAGS.pde == 'hyper_elasticity':
        err_left = coefs - ground_truth_vals
        mse_left = np.mean(err_left ** 2, axis=[1, 2]).reshape(-1, 1)

        points_right = np.array(points)
        points_right = points_right.at[:, :, 0].set(1. - points_right.at[:, :, 0].get())
        coefs_right = vmap(make_coef_func, (0, None, 0, 0))(
                keys, model, ground_truth_params, points_right
                )
        coefs_right = coefs_right.at[:, :, 0].multiply(-1.)
        err_right = coefs_right - ground_truth_vals
        mse_right = np.mean(err_right ** 2, axis=[1, 2]).reshape(-1, 1)

        def take_min(mse_left, mse_right, err_left, err_right):
            err, mse = jax.lax.cond(
                np.squeeze(mse_left) > np.squeeze(mse_right),
                lambda _: (err_right, mse_left),
                lambda _: (err_left, mse_left),
                None
            )
            return err, mse
        err, mse = jax.vmap(take_min, (0, 0, 0, 0))(mse_left, mse_right, err_left, err_right)
        mse = np.sum(mse)
        normalizer = np.mean(ground_truth_vals ** 2, axis=1, keepdims=True)
        rel_sq_err = err ** 2 / normalizer.mean(axis=2, keepdims=True)

    else:
        err = coefs - ground_truth_vals
        mse = np.mean(err ** 2)
        normalizer = np.mean(ground_truth_vals ** 2, axis=1, keepdims=True)
        rel_sq_err = err ** 2 / normalizer.mean(axis=2, keepdims=True)

    # if contains time dimension, add per time-stepping validation error
    if FLAGS.pde == 'td_burgers':
        assert points.shape[-1] == 2
        t_rel_sq_err = []
        tile_idx = (points.shape[1] // FLAGS.num_tsteps)
        t_idx = np.arange(0, tile_idx) * FLAGS.num_tsteps
        for i in range(FLAGS.num_tsteps):
            t_idx_i = t_idx + i
            #t_idx = np.arange(i * tile_idx, (i + 1) * tile_idx)
            t_err = err[:, t_idx_i, :]
            t_normalizer = np.mean(ground_truth_vals[:, t_idx_i, :] ** 2, axis=1, keepdims=True)
            t_rel_sq_err.append(np.mean(t_err ** 2 / t_normalizer.mean(axis=2, keepdims=True)))

    return (
        mse,
        np.mean(normalizer, axis=(0, 1)),
        np.mean(rel_sq_err),
        np.mean(rel_sq_err, axis=(0, 1)),
        np.std(np.mean(rel_sq_err, axis=(1, 2))),
        np.array(t_rel_sq_err) if FLAGS.pde == 'td_burgers' else None
    )


@partial(jax.jit, static_argnums=[4])
def vmap_validation_rollout(
    model, ground_truth_params, points, ground_truth_vals, make_coef_func
):
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, FLAGS.n_eval)

    coefs = vmap(make_coef_func, (0, None, 0, 0))(
        keys, model, ground_truth_params, points
    )
    return coefs


def get_optimizer(model_class, init_params):
    if FLAGS.optimizer == "adam":
        optimizer = flax.optim.Adam(learning_rate=FLAGS.outer_lr, beta1=0.9, beta2=0.99).create(
            flax.nn.Model(model_class, init_params)
        )
    elif FLAGS.optimizer == "rmsprop":
            optimizer = flax.optim.Adam(learning_rate=FLAGS.outer_lr, beta1=0., beta2=0.8).create(
                flax.nn.Model(model_class, init_params)
            )
    elif FLAGS.optimizer == "ranger":
        optimizer = flaxOptimizers.Ranger(
            learning_rate=FLAGS.outer_lr, beta2=0.99, use_gc=False
        ).create(flax.nn.Model(model_class, init_params))
    elif FLAGS.optimizer == "sgd":
        optimizer = flax.optim.Momentum(learning_rate=FLAGS.outer_lr, beta=0.0).create(
            flax.nn.Model(model_class, init_params))
    else:
        raise Exception("unknown optimizer: ", FLAGS.optimizer)

    return optimizer
