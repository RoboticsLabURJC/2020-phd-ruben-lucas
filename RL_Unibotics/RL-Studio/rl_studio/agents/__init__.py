from rl_studio.agents.agents_type import AgentsType
from rl_studio.agents.exceptions import NoValidTrainingType
from rl_studio.agents.tasks_type import TasksType
from rl_studio.agents.frameworks_type import FrameworksType
from rl_studio.agents.utils import print_messages
from rl_studio.algorithms.algorithms_type import AlgorithmsType
from rl_studio.envs.envs_type import EnvsType


class TrainerFactory:
    def __new__(cls, config):

        """
        There are many options:

         Tasks:
           - Follow_line
           - Follow_lane

         Agents:
           - F1 (Gazebo)
           - robot_mesh
           - Mountain car
           - Cartpole
           - Autoparking (Gazebo)
           - AutoCarla (Carla)
           - Turtlebot (Gazebo)

         Algorithms:
           - qlearn
           - DQN
           - DDPG
           - PPO
           - SAC

         Simulators:
           - Gazebo
           - OpenAI
           - Carla
           - SUMO
        """

        agent = config["settings"]["agent"]
        algorithm = config["settings"]["algorithm"]
        task = config["settings"].get("task")
        simulator = config["settings"].get("simulator")
        framework = config["settings"].get("framework")

        print_messages(
            "TrainerFactory",
            task=task,
            algorithm=algorithm,
            simulator=simulator,
            agent=agent,
            framework=framework,
        )
        # =============================
        # FollowLane - F1 - qlearn - Carla
        # =============================
        if (
            task == TasksType.FOLLOWLANECARLA.value
            and agent == AgentsType.AUTOCARLA.value
            and algorithm == AlgorithmsType.QLEARN.value
            and simulator == EnvsType.CARLA.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_qlearn_carla import (
                TrainerFollowLaneQlearnAutoCarla,
            )

            return TrainerFollowLaneQlearnAutoCarla(config)

        # =============================
        # FollowLine - F1 - qlearn - Gazebo
        # =============================
        if (
            task == TasksType.FOLLOWLINEGAZEBO.value
            and agent == AgentsType.F1GAZEBO.value
            and algorithm == AlgorithmsType.QLEARN.value
            and simulator == EnvsType.GAZEBO.value
        ):
            from rl_studio.agents.f1.train_followline_qlearn_f1_gazebo import (
                TrainerFollowLineQlearnF1Gazebo,
            )

            return TrainerFollowLineQlearnF1Gazebo(config)

        # =============================
        # FollowLine - F1 - DDPG - Gazebo - TensorFlow
        # =============================
        elif (
            task == TasksType.FOLLOWLINEGAZEBO.value
            and agent == AgentsType.F1GAZEBO.value
            and algorithm == AlgorithmsType.DDPG.value
            and simulator == EnvsType.GAZEBO.value
            and framework == FrameworksType.TF.value
        ):
            from rl_studio.agents.f1.train_followline_ddpg_f1_gazebo_tf import (
                TrainerFollowLineDDPGF1GazeboTF,
            )

            return TrainerFollowLineDDPGF1GazeboTF(config)

        # =============================
        # FollowLine - F1 - SAC - Gazebo - TensorFlow
        # =============================
        elif (
            task == TasksType.FOLLOWLINEGAZEBO.value
            and agent == AgentsType.F1GAZEBO.value
            and algorithm == AlgorithmsType.SAC.value
            and simulator == EnvsType.GAZEBO.value
            and framework == FrameworksType.TF.value
        ):
            from rl_studio.agents.f1.train_followline_sac_f1_gazebo_tf import (
                TrainerFollowLineSACF1GazeboTF,
            )

            return TrainerFollowLineSACF1GazeboTF(config)

        # =============================
        # FollowLine - F1 - PPO - Gazebo - TensorFlow
        # =============================
        elif (
                task == TasksType.FOLLOWLINEGAZEBO.value
                and agent == AgentsType.F1GAZEBO.value
                and algorithm == AlgorithmsType.PPO_CONTINIUOUS.value
                and simulator == EnvsType.GAZEBO.value
                and framework == FrameworksType.TF.value
        ):
            from rl_studio.agents.f1.train_followline_ppo_f1_gazebo_tf import (
                TrainerFollowLinePPOF1GazeboTF
            )

            return TrainerFollowLinePPOF1GazeboTF(config)

        # =============================
        # FollowLane - F1 - PPO - Carla
        # =============================
        elif (
            task == TasksType.FOLLOWLANECARLA.value
            and agent == AgentsType.AUTOCARLA.value
            and algorithm == AlgorithmsType.PPO_CONTINIUOUS.value
            and simulator == EnvsType.CARLA.value
            and framework == FrameworksType.TF
        ):
            from rl_studio.agents.auto_carla.train_followlane_ppo_carla import (
                TrainerFollowLanePPOCarla,
            )

            return TrainerFollowLanePPOCarla(config)

        elif (
            task == TasksType.FOLLOWLANECARLA.value
            and agent == AgentsType.F1GAZEBO.value
            and algorithm == AlgorithmsType.SAC.value
            and simulator == EnvsType.CARLA.value
            and framework == FrameworksType.TF.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_bs_sac_f1_carla import (
                TrainerFollowLaneSACF1CARLATF,
            )

            return TrainerFollowLaneSACF1CARLATF(config)
            return TrainerFollowLaneSACCarla(config)

        # =============================
        # FollowLane - F1 - SAC - Carla
        # =============================

        elif (
                task == TasksType.FOLLOWLANECARLA.value
                and agent == AgentsType.AUTOCARLA.value
                and algorithm == AlgorithmsType.SAC.value
                and simulator == EnvsType.CARLA.value
                and framework == FrameworksType.TF.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_sac_f1_carla_tf import (
                TrainerFollowLaneSACCarla,
            )

            return TrainerFollowLaneSACCarla(config)

        # =============================
        # FollowLane - F1 - DDPG BASELINES - Carla
        # =============================

        elif (
                task == TasksType.FOLLOWLANECARLA.value
                and agent == AgentsType.AUTOCARLA.value
                and algorithm == AlgorithmsType.DDPG.value
                and simulator == EnvsType.CARLA.value
                and framework == FrameworksType.BASELINES.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_bs_ddpg_f1_carla_tf import (
                TrainerFollowLaneDDPGCarla,
            )

            return TrainerFollowLaneDDPGCarla(config)

        elif (
                task == TasksType.FOLLOWLANECARLA.value
                and agent == AgentsType.AUTOCARLA.value
                and algorithm == AlgorithmsType.DDPG_2.value
                and simulator == EnvsType.CARLA.value
                and framework == FrameworksType.BASELINES_2.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_bs_ddpg_2_networks_f1_carla_tf import (
                TrainerFollowLaneDDPGCarla,
            )

            return TrainerFollowLaneDDPGCarla(config)


        elif (
                task == TasksType.FOLLOWLANECARLA.value
                and agent == AgentsType.AUTOCARLA.value
                and algorithm == AlgorithmsType.SAC_2.value
                and simulator == EnvsType.CARLA.value
                and framework == FrameworksType.BASELINES_2_SIMPLE.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_bs_sac_2_networks_f1_carla import (
                TrainerFollowLaneSACCarla,
            )

            return TrainerFollowLaneSACCarla(config)

        elif (
                task == TasksType.FOLLOWLANECARLA.value
                and agent == AgentsType.AUTOCARLA.value
                and algorithm == AlgorithmsType.PPO_CONTINIUOUS.value
                and simulator == EnvsType.CARLA.value
                and framework == FrameworksType.BASELINES.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_bs_ppo_f1_carla_tf import (
                TrainerFollowLanePPOCarla,
            )

            return TrainerFollowLanePPOCarla(config)
        elif (
                task == TasksType.FOLLOWLANECARLA.value
                and agent == AgentsType.AUTOCARLA.value
                and algorithm == AlgorithmsType.PPO_CONTINIUOUS.value
                and simulator == EnvsType.CARLA.value
                and framework == FrameworksType.PARALLEL_BASELINES.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_bs_parallel_ppo_f1_carla_tf import (
                TrainerFollowLanePPOCarla,
            )

            return TrainerFollowLanePPOCarla(config)
        elif (
                task == TasksType.FOLLOWLANECARLA.value
                and agent == AgentsType.AUTOCARLA.value
                and algorithm == AlgorithmsType.PPO_CONTINIUOUS.value
                and simulator == EnvsType.CARLA.value
                and framework == FrameworksType.TF.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_ppo_carla import (
                TrainerFollowLanePPOCarla,
            )

            return TrainerFollowLanePPOCarla(config)

        elif (
            task == TasksType.FOLLOWLANECARLA.value
            and agent == AgentsType.AUTOCARLA.value
            and algorithm == AlgorithmsType.MANUAL.value
            and simulator == EnvsType.CARLA.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_manual_carla import (
                TrainerFollowLaneManualCarla,
            )

            return TrainerFollowLaneManualCarla(config)
        elif (
            task == TasksType.FOLLOWLANECARLA.value
            and agent == AgentsType.AUTOCARLA.value
            and algorithm == AlgorithmsType.AUTO.value
            and simulator == EnvsType.CARLA.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_auto_carla import (
                TrainerFollowLaneAutoCarla,
            )

            return TrainerFollowLaneAutoCarla(config)
        # =============================
        # FollowLane - F1 - DDPG - Carla
        # =============================
        elif (
            task == TasksType.FOLLOWLANECARLA.value
            and agent == AgentsType.AUTOCARLA.value
            and algorithm == AlgorithmsType.DDPG.value
            and simulator == EnvsType.CARLA.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_ddpg_carla import (
                TrainerFollowLaneDDPGCarla,
            )

            return TrainerFollowLaneDDPGCarla(config)

        # =============================
        # FollowLine - F1 - DQN - Gazebo - TensorFlow
        # =============================
        elif (
            task == TasksType.FOLLOWLINEGAZEBO.value
            and agent == AgentsType.F1GAZEBO.value
            and algorithm == AlgorithmsType.DQN.value
            and simulator == EnvsType.GAZEBO.value
            and framework == FrameworksType.TF.value
        ):
            from rl_studio.agents.f1.train_followline_dqn_f1_gazebo_tf import (
                TrainerFollowLineDQNF1GazeboTF,
            )

            return TrainerFollowLineDQNF1GazeboTF(config)

        # =============================
        # Follow Lane - F1 - qlearn - Gazebo
        # =============================
        elif (
            task == TasksType.FOLLOWLANEGAZEBO.value
            and agent == AgentsType.F1GAZEBO.value
            and algorithm == AlgorithmsType.QLEARN.value
            and simulator == EnvsType.GAZEBO.value
        ):
            from rl_studio.agents.f1.train_followlane_qlearn_f1_gazebo import (
                TrainerFollowLaneQlearnF1Gazebo,
            )

            return TrainerFollowLaneQlearnF1Gazebo(config)

        # =============================
        # Follow Lane - F1 - DDPG - Gazebo - TF
        # =============================
        elif (
            task == TasksType.FOLLOWLANEGAZEBO.value
            and agent == AgentsType.F1GAZEBO.value
            and algorithm == AlgorithmsType.DDPG.value
            and simulator == EnvsType.GAZEBO.value
            and framework == FrameworksType.TF.value
        ):
            from rl_studio.agents.f1.train_followlane_ddpg_f1_gazebo_tf import (
                TrainerFollowLaneDDPGF1GazeboTF,
            )

            return TrainerFollowLaneDDPGF1GazeboTF(config)

        # =============================
        # Follow Lane - F1 - DQN - Gazebo - TF
        # =============================
        elif (
            task == TasksType.FOLLOWLANEGAZEBO.value
            and agent == AgentsType.F1GAZEBO.value
            and algorithm == AlgorithmsType.DQN.value
            and simulator == EnvsType.GAZEBO.value
            and framework == FrameworksType.TF.value
        ):
            from rl_studio.agents.f1.train_followlane_dqn_f1_gazebo_tf import (
                TrainerFollowLaneDQNF1GazeboTF,
            )

            return TrainerFollowLaneDQNF1GazeboTF(config)

        elif (
            task == TasksType.FOLLOWLANECARLA.value
            and agent == AgentsType.AUTOCARLA.value
            and algorithm == AlgorithmsType.SAC.value
            and simulator == EnvsType.CARLA.value
            and framework == FrameworksType.BASELINES.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_bs_sac_f1_carla import (
                TrainerFollowLaneSACCarla,
            )

            return TrainerFollowLaneSACCarla(config)

        # =============================
        # Robot Mesh - Qlearn - Gazebo
        # =============================
        elif (
            agent == AgentsType.ROBOT_MESH.value
            and algorithm == AlgorithmsType.QLEARN.value
        ):
            from rl_studio.agents.robot_mesh.train_qlearn import (
                QLearnRobotMeshTrainer as RobotMeshTrainer,
            )

            return RobotMeshTrainer(config)

        # =============================
        # Robot Mesh - Manual
        # =============================
        elif (
            agent == AgentsType.ROBOT_MESH.value
            and algorithm == AlgorithmsType.MANUAL.value
        ):
            from rl_studio.agents.robot_mesh.manual_pilot import (
                ManualRobotMeshTrainer as RobotMeshTrainer,
            )

            return RobotMeshTrainer(config)

        # =============================
        # Mountain Car - Qlearn
        # =============================
        elif (
            agent == AgentsType.MOUNTAIN_CAR.value
            and algorithm == AlgorithmsType.QLEARN.value
        ):
            from rl_studio.agents.mountain_car.train_qlearn import (
                QLearnMountainCarTrainer as MountainCarTrainer,
            )

            return MountainCarTrainer(config)

        # =============================
        # Mountain Car - Manual
        # =============================
        elif (
            agent == AgentsType.MOUNTAIN_CAR.value
            and algorithm == AlgorithmsType.MANUAL.value
        ):
            from rl_studio.agents.mountain_car.manual_pilot import (
                ManualMountainCarTrainerr as MountainCarTrainer,
            )

            return MountainCarTrainer(config)

        # =============================
        # CartPole - DQN
        # =============================
        elif (
            agent == AgentsType.CARTPOLE.value and algorithm == AlgorithmsType.DQN.value
        ):
            from rl_studio.agents.cartpole.train_dqn import (
                DQNCartpoleTrainer as CartpoleTrainer,
            )

            return CartpoleTrainer(config)

        # =============================
        # CartPole - Qlearn
        # =============================
        elif (
            agent == AgentsType.CARTPOLE.value
            and algorithm == AlgorithmsType.QLEARN.value
        ):
            from rl_studio.agents.cartpole.train_qlearn import (
                QLearnCartpoleTrainer as CartpoleTrainer,
            )

            return CartpoleTrainer(config)

        # =============================
        # CartPole - PPO
        # =============================
        elif (
            agent == AgentsType.CARTPOLE.value and algorithm == AlgorithmsType.PPO.value
        ):
            from rl_studio.agents.cartpole.train_ppo import (
                PPOCartpoleTrainer as CartpoleTrainer,
            )

            return CartpoleTrainer(config)
        # =============================
        # CartPole - PPO CONTINUOUS
        # =============================
        elif (
            agent == AgentsType.CARTPOLE.value
            and algorithm == AlgorithmsType.PPO_CONTINIUOUS.value
        ):
            from rl_studio.agents.cartpole.train_ppo_continous import (
                PPOCartpoleTrainer as CartpoleTrainer,
            )

            return CartpoleTrainer(config)

        # =============================
        # CartPole - DDPG
        # =============================
        elif (
            agent == AgentsType.CARTPOLE.value
            and algorithm == AlgorithmsType.DDPG.value
            and framework == FrameworksType.PYTORCH.value
        ):
            from rl_studio.agents.cartpole.train_ddpg import (
                DDPGCartpoleTrainer as CartpoleTrainer,
            )

            return CartpoleTrainer(config)

        # =============================
        # Autoparking - F1 - DDPG - Gazebo - TF
        # =============================
        elif (
            task == TasksType.AUTOPARKINGGAZEBO.value
            and agent == AgentsType.AUTOPARKINGGAZEBO.value
            and algorithm == AlgorithmsType.DDPG.value
            and simulator == EnvsType.GAZEBO.value
            and framework == FrameworksType.TF.value
        ):
            from rl_studio.agents.autoparking.train_ddpg import (
                TrainerAutoParkingDDPGGazeboTF,
            )

            return TrainerAutoParkingDDPGGazeboTF(config)

        # =============================
        # Autoparking - F1 - Qlearn - Gazebo
        # =============================
        elif (
            task == TasksType.AUTOPARKINGGAZEBO.value
            and agent == AgentsType.AUTOPARKINGGAZEBO.value
            and algorithm == AlgorithmsType.QLEARN.value
            and simulator == EnvsType.GAZEBO.value
        ):
            from rl_studio.agents.autoparking.train_ddpg import (
                TrainerAutoParkingQlearnGazebo,
            )

            return TrainerAutoParkingQlearnGazebo(config)

        # =============================
        # Turtlebot - Qlearn - Gazebo
        # =============================
        elif agent == AgentsType.TURTLEBOT.value:
            from rl_studio.agents.turtlebot.turtlebot_trainer import TurtlebotTrainer

            return TurtlebotTrainer(config)

        # =============================
        # Pendulum - DDPG - Pytorch
        # =============================
        elif agent == AgentsType.PENDULUM.value:
            if algorithm == AlgorithmsType.DDPG_TORCH.value:
                from rl_studio.agents.pendulum.train_ddpg import (
                    DDPGPendulumTrainer as PendulumTrainer,
                )

                return PendulumTrainer(config)

            elif algorithm == AlgorithmsType.PPO_CONTINIUOUS.value:
                from rl_studio.agents.pendulum.train_ppo import (
                    PPOPendulumTrainer as PendulumTrainer,
                )

                return PendulumTrainer(config)

        else:
            raise NoValidTrainingType(agent)


class InferencerFactory:
    def __new__(cls, config):

        agent = config["settings"]["agent"]
        algorithm = config["settings"]["algorithm"]
        task = config["settings"].get("task")
        simulator = config["settings"].get("simulator")
        framework = config["settings"].get("framework")
        print_messages(
            "InferenceExecutorFactory",
            task=task,
            algorithm=algorithm,
            simulator=simulator,
            agent=agent,
            framework=framework,
        )

        # =============================
        # FollowLine - F1 - qlearn - Gazebo
        # =============================
        if (
            task == TasksType.FOLLOWLINEGAZEBO.value
            and agent == AgentsType.F1GAZEBO.value
            and algorithm == AlgorithmsType.QLEARN.value
            and simulator == EnvsType.GAZEBO.value
        ):
            from rl_studio.agents.f1.inference_followline_qlearn_f1_gazebo import (
                InferencerFollowLineQlearnF1Gazebo,
            )

            return InferencerFollowLineQlearnF1Gazebo(config)

        # =============================
        # FollowLine - F1 - DDPG - Gazebo - TensorFlow
        # =============================
        elif (
            task == TasksType.FOLLOWLINEGAZEBO.value
            and agent == AgentsType.F1GAZEBO.value
            and algorithm == AlgorithmsType.DDPG.value
            and simulator == EnvsType.GAZEBO.value
            and framework == FrameworksType.TF.value
        ):
            from rl_studio.agents.f1.inference_followline_ddpg_f1_gazebo_tf import (
                InferencerFollowLineDDPGF1GazeboTF,
            )

            return InferencerFollowLineDDPGF1GazeboTF(config)

        # =============================
        # FollowLine - F1 - PPO - Gazebo - TF
        # =============================
        elif (
                task == TasksType.FOLLOWLINEGAZEBO.value
                and agent == AgentsType.F1GAZEBO.value
                and algorithm == AlgorithmsType.PPO_CONTINIUOUS.value
                and simulator == EnvsType.GAZEBO.value
                and framework == FrameworksType.TF.value
        ):
            from rl_studio.agents.f1.inference_ppo_continuous import (
                InferencerFollowLinePPOF1GazeboTF,
            )

            return InferencerFollowLinePPOF1GazeboTF(config)


        # =============================
        # FollowLine - F1 - DQN - Gazebo - TensorFlow
        # =============================
        elif (
            task == TasksType.FOLLOWLINEGAZEBO.value
            and agent == AgentsType.F1GAZEBO.value
            and algorithm == AlgorithmsType.DQN.value
            and simulator == EnvsType.GAZEBO.value
            and framework == FrameworksType.TF.value
        ):
            from rl_studio.agents.f1.inference_followline_dqn_f1_gazebo_tf import (
                InferencerFollowLineDQNF1GazeboTF,
            )

            return InferencerFollowLineDQNF1GazeboTF(config)

        # =============================
        # Follow Lane - F1 - qlearn - Gazebo
        # =============================
        elif (
            task == TasksType.FOLLOWLANEGAZEBO.value
            and agent == AgentsType.F1GAZEBO.value
            and algorithm == AlgorithmsType.QLEARN.value
            and simulator == EnvsType.GAZEBO.value
        ):
            from rl_studio.agents.f1.inference_followlane_qlearn_f1_gazebo import (
                InferencerFollowLaneQlearnF1Gazebo,
            )

            return InferencerFollowLaneQlearnF1Gazebo(config)

        # =============================
        # Follow Lane - F1 - DDPG - Gazebo - TF
        # =============================
        elif (
            task == TasksType.FOLLOWLANEGAZEBO.value
            and agent == AgentsType.F1GAZEBO.value
            and algorithm == AlgorithmsType.DDPG.value
            and simulator == EnvsType.GAZEBO.value
            and framework == FrameworksType.TF.value
        ):
            from rl_studio.agents.f1.inference_followlane_ddpg_f1_gazebo_tf import (
                InferencerFollowLaneDDPGF1GazeboTF,
            )

            return InferencerFollowLaneDDPGF1GazeboTF(config)

        # =============================
        # Follow Lane - F1 - DQN - Gazebo - TF
        # =============================
        elif (
            task == TasksType.FOLLOWLANEGAZEBO.value
            and agent == AgentsType.F1GAZEBO.value
            and algorithm == AlgorithmsType.DQN.value
            and simulator == EnvsType.GAZEBO.value
            and framework == FrameworksType.TF.value
        ):
            from rl_studio.agents.f1.inference_followlane_dqn_f1_gazebo_tf import (
                InferencerFollowLaneDQNF1GazeboTF,
            )

            return InferencerFollowLaneDQNF1GazeboTF(config)

        # =============================
        # Robot Mesh - Qlearn - Gazebo
        # =============================
        elif agent == AgentsType.ROBOT_MESH.value:
            from rl_studio.agents.robot_mesh.inference_qlearn import (
                QLearnRobotMeshInferencer,
            )

            return QLearnRobotMeshInferencer(config)
        elif (
            task == TasksType.FOLLOWLANECARLA.value
            and agent == AgentsType.AUTOCARLA.value
            and algorithm == AlgorithmsType.DDPG.value
            and simulator == EnvsType.CARLA.value
            and framework == FrameworksType.TF.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_ddpg_carla import (
                TrainerFollowLaneDDPGCarla,
            )

            return TrainerFollowLaneDDPGCarla(config)
        elif (
            task == TasksType.FOLLOWLANECARLA.value
            and agent == AgentsType.AUTOCARLA.value
            and algorithm == AlgorithmsType.DDPG.value
            and simulator == EnvsType.CARLA.value
            and framework == FrameworksType.BASELINES.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_bs_ddpg_f1_carla_tf import (
                TrainerFollowLaneDDPGCarla,
            )

            return TrainerFollowLaneDDPGCarla(config)

        elif (
                task == TasksType.FOLLOWLANECARLA.value
                and agent == AgentsType.AUTOCARLA.value
                and algorithm == AlgorithmsType.DDPG_2.value
                and simulator == EnvsType.CARLA.value
                and framework == FrameworksType.BASELINES_2.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_bs_ddpg_2_networks_f1_carla_tf import (
                TrainerFollowLaneDDPGCarla,
            )

            return TrainerFollowLaneDDPGCarla(config)


        elif (
                task == TasksType.FOLLOWLANECARLA.value
                and agent == AgentsType.AUTOCARLA.value
                and algorithm == AlgorithmsType.SAC_2.value
                and simulator == EnvsType.CARLA.value
                and framework == FrameworksType.BASELINES_2_SIMPLE.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_bs_sac_2_networks_f1_carla import (
                TrainerFollowLaneSACCarla,
            )

            return TrainerFollowLaneSACCarla(config)

        elif (
            task == TasksType.FOLLOWLANECARLA.value
            and agent == AgentsType.AUTOCARLA.value
            and algorithm == AlgorithmsType.PPO_CONTINIUOUS.value
            and simulator == EnvsType.CARLA.value
            and FrameworksType == FrameworksType.TF.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_ppo_carla import (
                TrainerFollowLanePPOCarla,
            )

            return TrainerFollowLanePPOCarla(config)

        elif (
            task == TasksType.FOLLOWLANECARLA.value
            and agent == AgentsType.AUTOCARLA.value
            and algorithm == AlgorithmsType.PPO_CONTINIUOUS.value
            and simulator == EnvsType.CARLA.value
            and framework == FrameworksType.PARALLEL_BASELINES.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_bs_parallel_ppo_f1_carla_tf import (
                TrainerFollowLanePPOCarla,
            )

            return TrainerFollowLanePPOCarla(config)
        elif (
                task == TasksType.FOLLOWLANECARLA.value
                and agent == AgentsType.AUTOCARLA.value
                and algorithm == AlgorithmsType.PPO_CONTINIUOUS.value
                and simulator == EnvsType.CARLA.value
                and framework == FrameworksType.BASELINES.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_bs_ppo_f1_carla_tf import (
                TrainerFollowLanePPOCarla,
            )

            return TrainerFollowLanePPOCarla(config)

        elif (
            task == TasksType.FOLLOWLANECARLA.value
            and agent == AgentsType.AUTOCARLA.value
            and algorithm == AlgorithmsType.SAC.value
            and simulator == EnvsType.CARLA.value
            and framework == FrameworksType.BASELINES.value
        ):
            from rl_studio.agents.auto_carla.train_followlane_bs_sac_f1_carla import (
                TrainerFollowLaneSACCarla,
            )

            return TrainerFollowLaneSACCarla(config)

        # =============================
        # CartPole
        # =============================
        elif agent == AgentsType.CARTPOLE.value:
            if algorithm == AlgorithmsType.DQN.value:
                from rl_studio.agents.cartpole.inference_dqn import (
                    DQNCartpoleInferencer as CartpoleInferencer,
                )
            elif algorithm == AlgorithmsType.PPO.value:
                from rl_studio.agents.cartpole.inference_ppo import (
                    PPOCartpoleInferencer as CartpoleInferencer,
                )
            elif algorithm == AlgorithmsType.PPO_CONTINIUOUS.value:
                from rl_studio.agents.cartpole.inference_ppo_continuous import (
                    PPOCartpoleInferencer as CartpoleInferencer,
                )
            elif (
                algorithm == AlgorithmsType.DDPG.value
                and framework == FrameworksType.PYTORCH.value
            ):
                from rl_studio.agents.cartpole.inference_ddpg import (
                    DDPGCartpoleInferencer as CartpoleInferencer,
                )
            elif algorithm == AlgorithmsType.PROGRAMMATIC.value:
                from rl_studio.agents.cartpole.inference_no_rl import (
                    NoRLCartpoleInferencer as CartpoleInferencer,
                )
            else:
                from rl_studio.agents.cartpole.inference_qlearn import (
                    QLearnCartpoleInferencer as CartpoleInferencer,
                )
            return CartpoleInferencer(config)

        # =============================
        # Mountain Car - Qlearn
        # =============================
        elif agent == AgentsType.MOUNTAIN_CAR.value:
            from rl_studio.agents.mountain_car.inference_qlearn import (
                QLearnMountainCarInferencer,
            )

            return QLearnMountainCarInferencer(config)

        # =============================
        # Pendulum - DDPG - Pytorch
        # =============================
        elif agent == AgentsType.PENDULUM.value:
            if algorithm == AlgorithmsType.DDPG_TORCH.value:
                from rl_studio.agents.pendulum.inference_ddpg import (
                    DDPGPendulumInferencer as PendulumInferencer,
                )

                return PendulumInferencer(config)

            elif algorithm == AlgorithmsType.PPO_CONTINIUOUS.value:
                from rl_studio.agents.pendulum.inference_ppo import (
                    PPOPendulumInferencer as PendulumInferencer,
                )

                return PendulumInferencer(config)
        else:
            raise NoValidTrainingType(agent)
