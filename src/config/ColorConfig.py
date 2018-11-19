# Copyright(c) Eric Steinberger 2018

from src.Color import Color
from src.config import RobotConfig

BLACK = Color(name="black", rgb=[25, 25, 25], position=RobotConfig.POS_COLOR_BLACK, color_id=0, code=[1, 1, 1, 0])
GREEN = Color(name="green", rgb=[0, 148, 84], position=RobotConfig.POS_COLOR_GREEN, color_id=1, code=[1, 0, 0, 0])
BLUE = Color(name="blue", rgb=[1, 51, 236], position=RobotConfig.POS_COLOR_BLUE, color_id=2, code=[1, 0, 1, 0])
YELLOW = Color(name="yellow", rgb=[255, 219, 0], position=RobotConfig.POS_COLOR_YELLOW, color_id=3, code=[0, 1, 0, 0])
RED = Color(name="red", rgb=[240, 14, 17], position=RobotConfig.POS_COLOR_RED, color_id=4, code=[0, 1, 1, 0])
WHITE = Color(name="white", rgb=[240, 240, 240], position=RobotConfig.POS_COLOR_WHITE, color_id=5, code=[0, 0, 1, 0])
ORANGE = Color(name="orange", rgb=[243, 125, 55], position=RobotConfig.POS_COLOR_ORANGE, color_id=6, code=[1, 1, 0, 0])
BROWN = Color(name="brown", rgb=[130, 54, 31], position=RobotConfig.POS_COLOR_BROWN, color_id=7, code=[0, 0, 0, 1])

ALL_COLORS_LIST = [BLACK, GREEN, BLUE, YELLOW, WHITE, RED, BROWN, ORANGE]
N_COLORS = len(ALL_COLORS_LIST)
