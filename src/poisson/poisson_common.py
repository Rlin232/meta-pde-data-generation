import jax
import jax.numpy as np
import numpy as npo
from jax import grad, jit, vmap
from ..nets.field import vmap_laplace_operator

from functools import partial
import flax
from flax import nn

import fenics as fa

import matplotlib.pyplot as plt
import pdb

from absl import app
from absl import flags

FLAGS = flags.FLAGS


def plot_solution(u, params=None):
    fa.plot(u)


def loss_fn(potential_fn, points, params):
    points_on_boundary, points_in_domain = points
    source_params, bc_params, _ = params

    err_on_boundary = vmap_boundary_conditions(
        points_on_boundary, bc_params
    ) - potential_fn(points_on_boundary)
    loss_on_boundary = np.mean(err_on_boundary ** 2)

    err_in_domain = vmap_laplace_operator(
        points_in_domain, potential_fn, lambda x: 1 + 0.1 * potential_fn(x) ** 2  # ) -
    ) - vmap_source(points_in_domain, source_params)
    loss_in_domain = np.mean(err_in_domain ** 2)
    return {"boundary_loss": loss_on_boundary}, {"domain_loss": loss_in_domain}


@jax.jit
def sample_params(key, seed=0):
    if FLAGS.fixed_num_pdes is not None:
        #key = jax.random.PRNGKey(
        #    jax.random.randint(
        #        key, (1,), np.array([0]), np.array([FLAGS.fixed_num_pdes])
        #    )[0]
        #)
        key = jax.random.PRNGKey(seed)

    k1, k2, k3 = jax.random.split(key, 3)

    # These keys will all be 0 if we're not varying that factor
    k1 = k1 * FLAGS.vary_source
    k2 = k2 * FLAGS.vary_bc
    k3 = k3 * FLAGS.vary_geometry

    source_params = jax.random.normal(k1, shape=(2, 3,))

    # bc_params = FLAGS.bc_scale * jax.random.uniform(
    #     k2, minval=0.0, maxval=0.0, shape=(5,)
    # )

    bc_params = FLAGS.bc_scale * jax.random.uniform(
        k2, minval=-1.0, maxval=1.0, shape=(5,)
    )

    geo_params = jax.random.uniform(k3, minval=-0.2, maxval=0.2, shape=(2,))

    return source_params, bc_params, geo_params


def sample_points(key, n, params):
    k1, k2 = jax.random.split(key)
    points_on_boundary = sample_points_on_boundary(k1, n, params)
    points_in_domain = sample_points_in_domain(k2, n, params)
    return (points_on_boundary, points_in_domain)


@partial(jax.jit, static_argnums=(1,))
def sample_points_on_boundary(key, n, params):
    _, _, geo_params = params
    c1, c2 = geo_params
    theta = np.linspace(0.0, 2 * np.pi, n)
    theta = theta + jax.random.uniform(
        key, minval=0.0, maxval=(2 * np.pi / n), shape=(n,)
    )
    r0 = 1.0 + c1 * np.cos(4 * theta) + c2 * np.cos(8 * theta)
    x = r0 * np.cos(theta)
    y = r0 * np.sin(theta)
    return np.stack([x, y], axis=1)


def is_in_hole(xy, geo_params, tol=1e-7):
    c1, c2 = geo_params
    vector = xy
    theta = np.arctan2(*vector)
    length = np.linalg.norm(vector)
    r0 = 1.0 + c1 * np.cos(4 * theta) + c2 * np.cos(8 * theta)
    return r0 < length + tol


@partial(jax.jit, static_argnums=(1,))
def sample_points_in_domain(key, n, params):
    _, _, geo_params = params
    k1, k2, k3 = jax.random.split(key, 3)
    # total number of points is 3 * n
    # so as long as the fraction of volume covered is << 1/3 we are ok
    n_x = 3 * n
    n_y = 3 * n

    xs = jax.random.uniform(k1, (n_x, ), minval=FLAGS.xmin, maxval=FLAGS.xmax)
    ys = jax.random.uniform(k2, (n_y, ), minval=FLAGS.ymin, maxval=FLAGS.ymax)

    xy = np.stack((xs.flatten(), ys.flatten()), axis=1)

    in_hole = jax.vmap(
        is_in_hole, in_axes=(0, None), out_axes=0
    )(xy, geo_params)

    idxs = jax.random.choice(k3, xy.shape[0], replace=False, p=1 - in_hole, shape=(n,))
    return xy[idxs]


@jax.jit
def boundary_conditions(r, x):
    """
    This returns the value required by the dirichlet boundary condition at x.
    """
    theta = np.arctan2(x[1], x[0])
    return (
        r[0]
        + r[1] / 4 * np.cos(theta)
        + r[2] / 4 * np.sin(theta)
        + r[3] / 4 * np.cos(2 * theta)
        + r[4] / 4 * np.sin(2 * theta)
    ).sum()


@jax.jit
def vmap_boundary_conditions(points_on_boundary, bc_params):
    return vmap(partial(boundary_conditions, bc_params))(points_on_boundary)


@jax.jit
def source(r, x):
    x = x.reshape([1, -1]) * np.ones([r.shape[0], x.shape[0]])
    results = r[:, 2] * np.exp(-((x[:, 0] - r[:, 0]) ** 2 + (x[:, 1] - r[:, 1]) ** 2))
    return results.sum()


@jax.jit
def vmap_source(points_in_domain, source_params):
    return vmap(partial(source, source_params))(points_in_domain)
