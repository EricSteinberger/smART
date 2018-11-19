# Copyright(c) Eric Steinberger 2018

"""
Interface for main() function creation in ABB RAPID code.
Calls previously created RAPID functions (see RapidFunctionWriter) as requested and writes that to self.rapid_file.
"""

import os

import numpy as np

from src import paths
from src.config import RobotConfig
from src.rapid.RapidAPI import RapidAPI
from src.util import file_util


class RapidFileWriter(object):
    def __init__(self, rapid_file, args):
        self.args = args
        self.rapid_file = rapid_file
        self.color_path_index = 0
        self.rapid_api = RapidAPI(args=args, rapid_file=rapid_file)

    def call_get_color(self, brush, color):
        z_off = 0
        self.rapid_file.write(
            "\t\t" + self.args.getcolor_func_name_prefix + color.capital + str(
                self.color_path_index % 3) + "_" + brush.name + " " + str(z_off) + ";" + "\n")
        self.color_path_index += 1

    def call_pump_clean(self, brush):
        self.rapid_file.write("\t\t" + self.args.clean_func_name_prefix + brush.name + ";\n")

    def call_towel(self, brush):
        # 2 random x offs
        r_x_1 = str(
            np.random.random() * (RobotConfig.POS_TOWEL_CornerLT[0] - RobotConfig.POS_TOWEL_CornerLB[0]))
        r_x_2 = str(
            np.random.random() * (RobotConfig.POS_TOWEL_CornerLT[0] - RobotConfig.POS_TOWEL_CornerLB[0]))

        # 2 random y offs
        r_y_1 = str(
            np.random.random() * (RobotConfig.POS_TOWEL_CornerRT[1] - RobotConfig.POS_TOWEL_CornerLT[1]))
        r_y_2 = str(
            np.random.random() * (RobotConfig.POS_TOWEL_CornerRT[1] - RobotConfig.POS_TOWEL_CornerLT[1]))

        self.rapid_file.write("\t\t" + self.args.towel_func_name_prefix + brush.name +
                              " " + r_x_1 + " ," + r_x_2 + " ," + r_y_1 + " ," + r_y_2 + ";\n")

    def call_brush_swap(self, from_brush, to_brush):
        self.rapid_file.write("\t\t" + self.args.change_brush_func_name_prefix
                              + "F_" + from_brush.name + "_TO_" + to_brush.name + ";\n")

    def call_activate_feedback_pin(self):
        if self.args.generate_io_calls_in_rapid:
            self.rapid_file.write("\t\t" + "SetDO do" + str(RobotConfig.feedback_pin_o) + ", 1;\n")

    def import_robtargets_from_all_rapid_fns(self):
        all_robtarget_files = file_util.get_all_txt_files_in_dir(paths.rapid_robtargets_path)
        for filename in all_robtarget_files:
            with open(os.path.join(paths.rapid_robtargets_path, filename)) as robt_RAPID_f:
                pseudo_rapid_code = robt_RAPID_f.read().splitlines()

                for loc in pseudo_rapid_code:
                    self.rapid_file.write("%s\n" % loc)

    def import_all_function_definitions(self):
        functions_path = paths.rapid_functions_path
        all_functiont_files = file_util.get_all_txt_files_in_dir(functions_path)
        for filename in all_functiont_files:
            with open(os.path.join(functions_path, filename)) as func_RAPID_f:
                fn_code = func_RAPID_f.read().splitlines()

                for loc in fn_code:
                    self.rapid_file.write("%s\n" % loc)

    def end_proc(self):
        self.rapid_api.write_end_proc()

    def end_module(self):
        self.rapid_api.write_end_module()

    def begin_module(self, name="Module1"):
        self.rapid_api.write_begin_module(name=name)
