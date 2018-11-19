# Copyright(c) Eric Steinberger 2018

"""
Interface to abstract ABB RAPID syntax away for function creation
"""

import os

from src import paths
from src.config import CleanConfig, RobotConfig


class RapidAPI:
    def __init__(self, args, rapid_file):
        self.args = args
        self.rapid_file = rapid_file

    def write_rapid_MoveL(self, pos8, point_name, pos_idx, fine=True, n_tabs=2):
        acc = "fine" if fine else "z0"

        rapid_cmd = "MoveL P" + point_name + "_" + str(int(pos_idx)) + ", v" + str(
            int(pos8[7])) + ", " + acc + ", pen\\WObj:=paintWO;"

        rapid_cmd = "\t" * n_tabs + rapid_cmd + "\n"
        self.rapid_file.write(rapid_cmd)
        return rapid_cmd

    def write_rapid_MoveL_with_offset(self, pos8, point_name, pos_idx,
                                      x_off_var_name="0", y_off_var_name="0", z_off_var_name="0",
                                      fine=True,
                                      n_tabs=2):

        acc = "fine" if fine else "z0"

        rapid_cmd = "MoveL Offs(P" + point_name + "_" + str(int(pos_idx)) \
                    + ", " + str(x_off_var_name) + ", " + str(y_off_var_name) + ", " + str(z_off_var_name) + ")," \
                    + " v" + str(int(pos8[7])) + ", " + acc + ", pen\\WObj:=paintWO;"

        rapid_cmd = "\t" * n_tabs + rapid_cmd + "\n"

        self.rapid_file.write(rapid_cmd)
        return rapid_cmd

    def write_rapid_Wait(self, n_milliseconds, n_tabs=2):
        rapid_cmd = "\t" * n_tabs + "WaitTime " + str(n_milliseconds) + ";\n"
        self.rapid_file.write(rapid_cmd)
        return rapid_cmd

    def write_open_hydraulic(self, n_tabs=2):
        if self.args.generate_io_calls_in_rapid:
            rapid_cmd = "\t" * n_tabs + "SetDO do" + str(RobotConfig.hydraulic_io_pin) + ", 1;\n"
            self.rapid_file.write(rapid_cmd)
            return rapid_cmd

    def write_close_hydraulic(self, n_tabs=2):
        if self.args.generate_io_calls_in_rapid:
            rapid_cmd = "\t" * n_tabs + "SetDO do" + str(RobotConfig.hydraulic_io_pin) + ", 0;\n"
            self.rapid_file.write(rapid_cmd)
            return rapid_cmd

    def write_start_water(self, n_tabs=2):
        if self.args.generate_io_calls_in_rapid:
            rapid_cmd = ""
            for i in range(len(CleanConfig.code)):
                rapid_cmd += "\t" * n_tabs + "SetDo do" + str(RobotConfig.out_pins[i]) + ", " + str(
                    CleanConfig.code[i]) + ";\n"

            rapid_cmd += "\t" * n_tabs + "WHILE DInput(di" + str(RobotConfig.feedback_pin_i) + ") = 0 DO\n"
            rapid_cmd += "\t" * (n_tabs + 1) + "WaitTime 0.01;\n"
            rapid_cmd += "\t" * n_tabs + "ENDWHILE\n\n"

            self.rapid_file.write(rapid_cmd)
            return rapid_cmd

    def write_stop_water(self, n_tabs=2):
        if self.args.generate_io_calls_in_rapid:
            rapid_cmd = ""
            for i in range(len(CleanConfig.code)):
                rapid_cmd += "\t" * n_tabs + "SetDo do" + str(RobotConfig.out_pins[i]) + ", 0;\n\n"

            self.rapid_file.write(rapid_cmd)
            return rapid_cmd

    def write_open_bucket(self, color, n_tabs=2):
        if self.args.generate_io_calls_in_rapid:
            rapid_cmd = ""
            for i in range(len(color.code)):
                rapid_cmd += "\t" * n_tabs + "SetDo do" + str(RobotConfig.out_pins[i]) + ", " + str(
                    color.code[i]) + ";\n"

            rapid_cmd += "\t" * n_tabs + "WHILE DInput(di" + str(RobotConfig.feedback_pin_i) + ") = 0 DO\n"
            rapid_cmd += "\t" * (n_tabs + 1) + "WaitTime 0.01;\n"
            rapid_cmd += "\t" * n_tabs + "ENDWHILE\n"

            self.rapid_file.write(rapid_cmd)
            return rapid_cmd

    def write_close_bucket(self, color, n_tabs=2):
        if self.args.generate_io_calls_in_rapid:
            rapid_cmd = ""
            for i in range(len(color.code)):
                rapid_cmd += "\t" * n_tabs + "SetDo do" + str(RobotConfig.out_pins[i]) + ", 0;\n"

            # rapid_cmd += "\t" * n_tabs + "WaitTime 0.5;\n"
            rapid_cmd += "\t" * n_tabs + "WHILE DInput(di" + str(RobotConfig.feedback_pin_i) + ") = 0 DO\n"
            rapid_cmd += "\t" * (n_tabs + 1) + "WaitTime 0.01;\n"
            rapid_cmd += "\t" * n_tabs + "ENDWHILE\n"

            self.rapid_file.write(rapid_cmd)
            return rapid_cmd

    def write_mount_brush(self, to_brush, n_tabs=2):
        self.rapid_file.write("\t" * n_tabs + self.args.mount_brush_func_name_prefix + to_brush.name + ";\n")

    def write_ummount_brush(self, from_brush, n_tabs=2):
        rapid_cmd = "\t" * n_tabs + self.args.unmount_brush_func_name_prefix + from_brush.name + ";\n"
        self.rapid_file.write(rapid_cmd)
        return rapid_cmd

    def write_proc(self, fn_name, arg_names_list=None, n_tabs=1, n_newlines=2):
        rapid_cmd = "\n" * n_newlines + "\t" * n_tabs + "PROC " + fn_name + "("
        one_already_written = False

        if arg_names_list is not None:
            for e in arg_names_list:
                if one_already_written:
                    rapid_cmd += ", "
                rapid_cmd += "num " + str(e)

                one_already_written = True

        rapid_cmd += ")\n"

        self.rapid_file.write(rapid_cmd)
        return rapid_cmd

    def write_end_proc(self):
        rapid_cmd = "\tENDPROC\n"
        self.rapid_file.write(rapid_cmd)
        return rapid_cmd

    def write_end_module(self):
        rapid_cmd = "ENDMODULE\n"
        self.rapid_file.write(rapid_cmd)
        return rapid_cmd

    def write_begin_module(self, name="Module1"):
        rapid_cmd = "MODULE " + name + "\n\n"
        self.rapid_file.write(rapid_cmd)
        return rapid_cmd

    @staticmethod
    def make_robtargets_file(movements, func_name):
        with open(os.path.join(paths.rapid_robtargets_path, func_name + "_Points.txt"), 'w') as rapid_file:
            for idx, pos8 in enumerate(movements):
                px = float(pos8[0])
                py = float(pos8[1])
                pz = float(pos8[2])
                rw = float(pos8[3])
                rx = float(pos8[4])
                ry = float(pos8[5])
                rz = float(pos8[6])
                px = str("{0:.7f}".format(round(px, 7)))
                py = str("{0:.7f}".format(round(py, 7)))
                pz = str("{0:.7f}".format(round(pz, 7)))
                rw = str("{0:.7f}".format(round(rw, 7)))
                rx = str("{0:.7f}".format(round(rx, 7)))
                ry = str("{0:.7f}".format(round(ry, 7)))
                rz = str("{0:.7f}".format(round(rz, 7)))

                robtarget = "\tCONST robtarget P" + func_name + "_" + str(idx) \
                            + ":=[[" + px + "," + py + "," + pz + "],[" + rw + "," + rx + "," + ry + "," + rz + "]," \
                            + "[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];"

                rapid_file.write(robtarget + "\n")
