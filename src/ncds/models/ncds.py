import jax
import jax.numpy as jnp
from dataclasses import dataclass
from jax import value_and_grad
from jax._src.typing import Array
from typing import Any, Callable, Dict
import flax.linen as nn
from jax.tree_util import tree_map, tree_reduce
from tqdm import tqdm
from flax.training import train_state, checkpoints
import optax
import torch
import numpy as np


@dataclass
class GNetConfig:
    num_layers: int
    hidden_dim: int
    output_dim: int

@dataclass
class JNetConfig:
    g_net_config: GNetConfig
    epsilon_init: float = 1e-6
    train_epsilon: bool = False

@dataclass
class FNetConfig:
    j_net_config: JNetConfig
    x0_init: Array
    train_x0: bool = False
    K: int = 20

@dataclass
class OptimConfig:
    learning_rate: float

@dataclass
class LossWeights:
    reconstruction: float = 1.0

@dataclass
class NCDSConfig:
    f_net_config: FNetConfig
    optimizer: OptimConfig
    loss_weights: LossWeights
    seed: int = 42


class TrainState(train_state.TrainState):
    weights: Dict


class NCDS:
    
    def __init__(self, ncds_config: NCDSConfig):
        
        self.config = ncds_config
        
        self.model = F_net(**vars(self.config.f_net_config))
        self.create_functions()    
        
    def create_functions(self):
        
        def losses(params, batch, dt):
            velocity = (batch[:, 1:] - batch[:, :-1]) / dt
            
            
            pred = jax.vmap(self.model.apply, in_axes=(None, 0))(params, batch[:, :-1]).squeeze(-1)
            
            # pred = self.model.apply(params, batch[:, :-1])
            
            velocity_loss = jnp.mean(
                jnp.sum((pred - velocity)**2, axis=-1)
            )
            
            return {'reconstruction': velocity_loss}
    
        def loss_fn(params, batch, weights, dt):
            all_losses = losses(params, batch, dt)
            weighted_losses = tree_map(lambda x, y: x*y, all_losses, weights) 
            loss = tree_reduce(lambda x, y: x + y, weighted_losses)
            return loss
            
        def step(state, batch, dt):
            
            loss, grads = value_and_grad(
                loss_fn, has_aux=False
            )(state.params, batch, state.weights, dt)
        
            state = state.apply_gradients(grads=grads)
            return loss, state
        self.losses = losses
        self.loss_fn = loss_fn
        self.step = jax.jit(step)
    
    def init_model(self, init_example: Array) -> Any:
        
        self.key = jax.random.PRNGKey(self.config.seed)
        self.key, init_subkey = jax.random.split(self.key)
        
        model_params = self.model.init(init_subkey, init_example)
        weights = vars(self.config.loss_weights)
        tx = optax.adam(self.config.optimizer.learning_rate)
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=model_params,
            tx=tx,
            weights=weights
        )
    
    def train_model(self, dataloader: Any, dt: float, num_epochs: int = 2000) -> None:
        if not hasattr(self, 'state'):
            if isinstance(dataloader, torch.utils.data.DataLoader):
                init_example = next(iter(dataloader))[0]
            elif isinstance(dataloader, Array):
                init_example = dataloader[0][0]
            elif isinstance(dataloader, np.ndarray):
                init_example = jnp.asarray(dataloader[0][0])
            else:
                raise ValueError("dataloader must be a DataLoader or a JAX Array")
            
            self.init_model(init_example)
        for epoch in range(num_epochs):
            epoch_loss = self.train_epoch(dataloader, dt)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
    
    def train_epoch(self, dataloader: Any, dt: float) -> float:
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader):
            batch_loss, self.state = self.step(self.state, batch, dt)
            epoch_loss += batch_loss
            num_batches += 1
        
        epoch_loss /= num_batches
        return epoch_loss

    @jax.jit(static_argnums=0)
    def model_fn(self, x: Array) -> Array:
        return self.model.apply(self.state.params, x)

    def save_checkpoint(self, step: int) -> None:
        checkpoints.save_checkpoint(
            ckpt_dir=self.config.checkpoint_dir,
            target=self.state,
            step=step,
            overwrite=True,
        )
    
    def load_checkpoint(self, step: int) -> None:
        self.state = checkpoints.restore_checkpoint(
            ckpt_dir=self.config.checkpoint_dir,
            target=self.state,
            step=step,
        )


class G_net(nn.Module):
    
    num_layers: int
    hidden_dim: int
    output_dim: int
    activation: Callable[[Array], Array] = nn.tanh
    
    @nn.compact

    def __call__(self, x: Array) -> Array:
        
        for i in range(self.num_layers-1):
            x = nn.Dense(
                self.hidden_dim,
                # Call the factory to obtain the initializer callable.
                kernel_init=jax.nn.initializers.glorot_normal(),
            )(x)
            x = self.activation(x)
        
        x = nn.Dense(
            self.output_dim**2,
            kernel_init=jax.nn.initializers.glorot_normal(),
        )(x)
        
        return x.reshape(x.shape[0], self.output_dim, self.output_dim)
    

class J_net(nn.Module):
    
    g_net_config: GNetConfig
    epsilon_init: float = 1e-6
    train_epsilon: bool = False
    
    def setup(self):
        self.g_net = G_net(**vars(self.g_net_config))
        self.epsilon = self.epsilon_init if not self.train_epsilon else self.param("epsilon", lambda _: self.epsilon_init)
    
    def __call__(self, x: Array) -> Array:
        g_out = self.g_net(x)
        j_out = - (g_out @ jnp.transpose(g_out, (0, 2, 1))) + self.epsilon * jnp.eye(g_out.shape[-1])
        return j_out


class F_net(nn.Module):
    
    j_net_config: JNetConfig
    x0_init: Array
    train_x0: bool = False
    K: int = 20
    
    def setup(self):
        self.j_net = J_net(**vars(self.j_net_config))
        self.x0 = self.x0_init if not self.train_x0 else self.param("x0", lambda _: self.x0_init)

    def __call__(self, x: Array) -> Array:
        """
        f(x) = f(x0) + \int J_f(x0 + tau (x - x0))(x - x0) dtau
        with tau in [0, 1]
        """
        
        direction = x - self.x0
        acc = 0.0
        for k in range(1, self.K + 1):
            tau = k / self.K
            acc += self.j_net(self.x0 + tau * direction) @ direction[..., None]
        
        return acc / self.K

    
    
    
