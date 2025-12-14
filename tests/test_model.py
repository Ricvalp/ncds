import jax.numpy as jnp
import jax

from ncds.models.ncds import (
    GNetConfig,
    JNetConfig,
    FNetConfig,
    G_net,
    J_net,
    F_net,
)


def test_model():
    
    g_net_config = GNetConfig(
        num_layers=3,
        hidden_dim=128,
        output_dim=2,
    )
    
    j_net_config = JNetConfig(
        g_net_config=g_net_config,
        epsilon_init=1e-6,
        train_epsilon=True,
    )
    
    f_net_config = FNetConfig(
        j_net_config=j_net_config,
        x0_init=jnp.zeros((2,)),
        train_x0=True,
        K=20,
    )
    
    key = jax.random.PRNGKey(42)
    key, data_subkey, init_subkey = jax.random.split(key, 3)
    
    dummy_data = jax.random.normal(data_subkey, (100, 2))
    
    g_net = G_net(**vars(g_net_config))
    g_net_params = g_net.init(init_subkey, dummy_data)
    g_net_output = g_net.apply(g_net_params, dummy_data)
    
    assert g_net_output.shape == (100, 2, 2)
    
    key, init_subkey = jax.random.split(key)
    j_net = J_net(**vars(j_net_config))
    j_net_params = j_net.init(init_subkey, dummy_data)
    j_net_output = j_net.apply(j_net_params, dummy_data)
    
    assert j_net_output.shape == (100, 2, 2)
    
    key, init_subkey = jax.random.split(key)
    f_net = F_net(**vars(f_net_config))
    f_net_params = f_net.init(init_subkey, dummy_data)
    f_net_output = f_net.apply(f_net_params, dummy_data)
    
    assert f_net_output.shape == (100, 2, 1)
