from rl_studio.agents.auto_carla.base_auto_carla_trainer import BaseAutoCarlaTrainer
from rl_studio.agents.factories.abstract_trainer_factory import TrainerFactory

class TrainACCAutoCarla(BaseAutoCarlaTrainer):
    def __init__(self, config: dict):
        super().__init__(config)

    def setup_env(self):
        from rl_studio.envs.carla.acc.acc_carla_sb_3 import FollowLaneStaticWeatherNoTraffic
        self.env = FollowLaneStaticWeatherNoTraffic(**self.config["env_params"])
        self.env.reset()
        print("env setup")

@TrainerFactory.register("train_acc_auto_carla")
def train_acc_auto_carla(config: dict):
    return TrainACCAutoCarla(config=config)
