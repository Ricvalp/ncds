import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from ncds.data.lasa import load_lasa_shape
from ncds.models.ncds import (
    NCDS,
    FNetConfig,
    GNetConfig,
    JNetConfig,
    LossWeights,
    NCDSConfig,
    OptimConfig,
)
from ncds.models.utils import plot_solution_trajectories, plot_vector_field


def train(config: ConfigDict) -> None:

    key = jax.random.PRNGKey(config.seed)
    key, _ = jax.random.split(key)

    g_net_config = GNetConfig(
        num_layers=config.g_net.num_layers,
        hidden_dim=config.g_net.hidden_dim,
        output_dim=config.g_net.output_dim,
    )

    j_net_config = JNetConfig(
        g_net_config=g_net_config,
        epsilon_init=config.j_net.epsilon_init,
        train_epsilon=config.j_net.train_epsilon,
    )

    f_net_config = FNetConfig(
        j_net_config=j_net_config,
        x0_init=jnp.asarray(config.f_net.x0_init),
        train_x0=config.f_net.train_x0,
        K=config.f_net.K,
    )

    ncds_config = NCDSConfig(
        f_net_config=f_net_config,
        optimizer=OptimConfig(learning_rate=config.optimizer.learning_rate),
        loss_weights=LossWeights(reconstruction=config.loss_weights.reconstruction),
        seed=config.seed,
    )

    data = load_lasa_shape(
        "dataset/lasahandwritingdataset/DataSet/CShape.mat",
        normalize=True,
    )

    data_trajectories = jnp.array(data["trajectories"])
    data_iterator = jnp.tile(data_trajectories, (128, 1, 1, 1))
    dt = data["dt"]

    ncds = NCDS(ncds_config)
    ncds.train_model(
        dataloader=data_iterator,
        dt=dt,
        num_epochs=config.training.num_epochs,
    )

    if config.plot_results:

        x = jnp.linspace(-2.5, 0.5, 10)
        y = jnp.linspace(-0.5, 2.5, 10)
        X, Y = jnp.meshgrid(x, y)
        initial_points = jnp.vstack([X.ravel(), Y.ravel()]).T

        solution_trajectories = integrate_ncds(
            initial_points=initial_points,
            ncds=ncds,
            dt=dt,
            num_steps=1000,
        )

        plot_vector_field(
            ncds.model_fn,
            training_traj=data_trajectories,
            xlim=(-2.5, 0.5),
            ylim=(-0.5, 2.5),
            grid_size=20,
            save_path="figures/trained_vector_field.png",
        )

        plot_solution_trajectories(
            trajectories=solution_trajectories,
            title="NCDS Solution Trajectories",
            save_path="figures/solution_trajectories.png",
        )


def integrate_ncds(
    initial_points: jnp.ndarray, ncds: NCDS, dt: float, num_steps: int
) -> jnp.ndarray:

    trajectory = [initial_points]
    current_points = initial_points
    for _ in range(num_steps):
        velocities = ncds.model_fn(current_points).squeeze(-1)
        current_points = current_points + velocities * dt
        trajectory.append(current_points)
    trajectory = jnp.stack(trajectory, axis=1)
    return trajectory
