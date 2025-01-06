from enum import Enum


class TasksType(Enum):
    FOLLOWLINEGAZEBO = "follow_line_gazebo"
    FOLLOWLANEGAZEBO = "follow_lane_gazebo"
    FOLLOWLANECARLA = "follow_lane_carla"
    FOLLOWLANECARLA8 = "follow_lane_carla_8"
    AUTOPARKINGGAZEBO = "autoparking_gazebo"
