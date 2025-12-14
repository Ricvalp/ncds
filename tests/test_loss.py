import jax
import jax.numpy as jnp

from ncds.models.ncds import (
    NCDS,
    FNetConfig,
    GNetConfig,
    JNetConfig,
    LossWeights,
    NCDSConfig,
    OptimConfig,
)


def test_loss():
    ncds_config = NCDSConfig(
        f_net_config=FNetConfig(
            j_net_config=JNetConfig(
                g_net_config=GNetConfig(
                    num_layers=2,
                    hidden_dim=64,
                    output_dim=2,
                ),
                epsilon_init=1e-6,
                train_epsilon=True,
            ),
            x0_init=jnp.zeros((2,)),
            train_x0=True,
            K=10,
        ),
        optimizer=OptimConfig(learning_rate=1e-3),
        loss_weights=LossWeights(reconstruction=1.0),
        seed=42,
    )

    ncds = NCDS(ncds_config)

    key = jax.random.PRNGKey(0)
    key, data_subkey = jax.random.split(key)

    dummy_batch = jax.random.normal(data_subkey, (100, 2))
    ncds.init_model(dummy_batch)
    dt = 0.1

    dummy_dataloader = jnp.tile(dummy_batch[None, None, ...], (3, 1, 1, 1))
    ncds.train_epoch(dummy_dataloader, dt)
