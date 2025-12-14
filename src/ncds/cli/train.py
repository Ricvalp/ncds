from ml_collections import config_flags

from ncds.training.train import train

_CONFIG_FILE = config_flags.DEFINE_config_file("config", default="configs/train.py")


def load_config(_CONFIG_FILE):
    cfg = _CONFIG_FILE.value
    return cfg


def main(_):
    config = load_config(_CONFIG_FILE)
    train(config)


if __name__ == "__main__":
    from absl import app

    app.run(main)
