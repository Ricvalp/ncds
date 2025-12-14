from ml_collections import ConfigDict

def get_config() -> ConfigDict:
    
    cfg = ConfigDict()
    
    cfg.seed = 42
    cfg.checkpoint_dir = "checkpoints/"
    cfg.plot_results = True
    
    cfg.g_net = ConfigDict(
        dict(
            num_layers=3,
            hidden_dim=128,
            output_dim=2,
        )
    )
    
    cfg.j_net = ConfigDict(
        dict(
            epsilon_init=1e-6,
            train_epsilon=True,
        )
    )
    
    cfg.f_net = ConfigDict(
        dict(
            x0_init=[0.0, 0.0],
            train_x0=True,
            K=20,
        )
    )
    
    cfg.optimizer = ConfigDict(dict(learning_rate=1e-3))
    cfg.loss_weights = ConfigDict(dict(reconstruction=1.0))
    
    cfg.training = ConfigDict(
        dict(
            dt=0.01,
            num_epochs=100,
        )
    )
    
    return cfg
