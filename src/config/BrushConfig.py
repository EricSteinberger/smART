# Copyright(c) Eric Steinberger 2018

from src.Brush import Brush
from src.config import RobotConfig


NOTHING_MOUNTED = Brush(name="NOTHING_MOUNTED",
                        height=0,
                        reference_height=RobotConfig.RS_TOOL_BRUSH_Z,
                        delta_z_paint=0,
                        delta_z_towel=0,
                        delta_z_color=0,
                        delta_z_water=0,
                        position_for_pick_up=RobotConfig.POS_BRUSH_NOTHING_MOUNTED,
                        padding_pickup_z=200,
                        stroke_names_list=[])

B6 = Brush(name="B6",
           height=190.0 - 4.3,
           reference_height=RobotConfig.RS_TOOL_BRUSH_Z,
           delta_z_paint=0,
           delta_z_towel=-1.3,
           delta_z_color=1,
           delta_z_water=-3,
           position_for_pick_up=RobotConfig.POS_BRUSH_STATION_2,
           padding_pickup_z=200,
           stroke_names_list=["B6_0", "B6_1", "B6_2"])

B12 = Brush(name="B12",
            height=199.6,
            reference_height=RobotConfig.RS_TOOL_BRUSH_Z,
            delta_z_paint=0,
            delta_z_towel=-12.2,
            delta_z_color=-2.8,
            delta_z_water=-14.0,
            position_for_pick_up=RobotConfig.POS_BRUSH_STATION_3,
            padding_pickup_z=200,
            stroke_names_list=["B12_0", "B12_1", "B12_2", "B12_3", "B12_4"])

N_BRUSHES = 2
ALL_BRUSHES_LIST = [B6, B12, NOTHING_MOUNTED]
ALL_PAINTING_BRUSHES_LIST = [B6, B12]
BRUSH_AT_BOOT = NOTHING_MOUNTED
