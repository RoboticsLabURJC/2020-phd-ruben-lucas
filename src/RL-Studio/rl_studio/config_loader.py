import yaml
import os
from rl_studio.agents.f1.loaders import load_common_env_params

CONFIG_PATH = None  # Global variable

def set_config_path(path):
    global CONFIG_PATH
    CONFIG_PATH = os.path.abspath(path)


def load_config():
    if CONFIG_PATH is None:
        raise ValueError("CONFIG_PATH not set")

    with open(CONFIG_PATH, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_hot_config():
    if CONFIG_PATH is None:
        raise ValueError("CONFIG_PATH not set")

    with open(CONFIG_PATH, 'r') as f:
        config_file = yaml.load(f, Loader=yaml.FullLoader)

    environment = {}
    load_common_env_params(environment, config_file)
    return environment