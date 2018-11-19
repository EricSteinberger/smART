# Copyright(c) Eric Steinberger 2018

"""
Interface to write functions in ABB RAPID code that can later be called in the RAPID main function
"""

import copy
import os

import numpy as np

from src import paths
from src.config import BrushConfig, ColorConfig, RobotConfig
from src.rapid.RapidAPI import RapidAPI
from src.simulation.stroke_tools import get_n_circles
from src.simulation.stroke_tools import pad_strokes_with_lift_up, get_centered_stroke, \
    create_rotated_versions_of_stroke, offset_frame_by_x_y_z
from src.util import file_util


class RapidFunctionWriter:
    def __init__(self, args):
        self.args = args

    def write_all(self):
        # all support functions for all brushes
        self.write_rapid_all_brush_mount_and_unmount_fns()
        self.write_rapid_all_brush_swap_fns()
        self.write_rapid_pump_clean_for_all_brushes()
        self.write_rapid_get_color_fns_for_all_brushes()
        self.write_rapid_towel_for_all_brushes()

        # strokes
        self.write_rapid_all_stroke_functions()

    def write_rapid_all_stroke_functions(self):
        for brush in BrushConfig.ALL_PAINTING_BRUSHES_LIST:
            for stroke_id, stroke_name in enumerate(brush.stroke_names_list):
                print("Generating stroke rapid function:", stroke_name)

                # load the stroke's movement array
                stroke_np_path = os.path.join(paths.stroke_movements_npy_path, brush.name, stroke_name + ".npy")
                stroke_np = np.load(stroke_np_path)
                stroke_np = get_centered_stroke(stroke_np)

                # create rotated versions of the stroke
                rotated_strokes_list = create_rotated_versions_of_stroke(
                    stroke=stroke_np, num_rotations=self.args.n_stroke_rotations)
                rotated_strokes_list = pad_strokes_with_lift_up(list_of_strokes=rotated_strokes_list,
                                                                z_pad_mm=RobotConfig.PAINT_SAFETY_PADDING_Z)

                # adjust robot speed for strokes
                for s in range(self.args.n_stroke_rotations):
                    for i in range(1, len(rotated_strokes_list[s])):
                        rotated_strokes_list[s][i][7] = RobotConfig.SPEED_STROKE
                    rotated_strokes_list[s][0][7] = RobotConfig.SPEED_TRAVEL

                # write RAPID code
                for rotation_id in range(self.args.n_stroke_rotations):
                    self._write_stroke_fn(stroke_name=stroke_name + "_" + str(rotation_id + 1),
                                          stroke_movements=rotated_strokes_list[rotation_id])

    def write_rapid_towel_for_all_brushes(self):
        Q = RobotConfig.STANDARD_QUATERNIONS
        for brush in BrushConfig.ALL_BRUSHES_LIST:
            brush_func_name = self.args.towel_func_name_prefix + brush.name
            base_movements = []
            save_position = RobotConfig.POS_TOWEL_TRANSITION_POINT + Q + [RobotConfig.SPEED_TRAVEL]

            z_internal_trans = 15

            lb = copy.deepcopy(RobotConfig.POS_TOWEL_CornerLB)
            lb[2] += brush.delta_z_towel
            rb = copy.deepcopy(RobotConfig.POS_TOWEL_CornerRB)
            rb[2] += brush.delta_z_towel
            lt = copy.deepcopy(RobotConfig.POS_TOWEL_CornerLT)
            lt[2] += brush.delta_z_towel

            # 0-4 optimization_image slide
            base_movements.append(
                offset_frame_by_x_y_z(lb + Q + [RobotConfig.SPEED_TRAVEL], z=z_internal_trans))
            base_movements.append(lb + Q + [RobotConfig.SPEED_CLEAN])
            base_movements.append(rb + Q + [RobotConfig.SPEED_CLEAN])
            base_movements.append(
                offset_frame_by_x_y_z(rb + Q + [RobotConfig.SPEED_CLEAN], z=z_internal_trans))

            # 5-8 optimization_image slide
            base_movements.append(
                offset_frame_by_x_y_z(rb + Q + [RobotConfig.SPEED_TRAVEL], z=z_internal_trans))
            base_movements.append(rb + Q + [RobotConfig.SPEED_CLEAN])
            base_movements.append(lb + Q + [RobotConfig.SPEED_CLEAN])
            base_movements.append(
                offset_frame_by_x_y_z(lb + Q + [RobotConfig.SPEED_CLEAN], z=z_internal_trans))

            # y slide
            base_movements.append(
                offset_frame_by_x_y_z(lb + Q + [RobotConfig.SPEED_TRAVEL], z=z_internal_trans))
            base_movements.append(lb + Q + [RobotConfig.SPEED_CLEAN])
            base_movements.append(lt + Q + [RobotConfig.SPEED_CLEAN])
            base_movements.append(
                offset_frame_by_x_y_z(lt + Q + [RobotConfig.SPEED_CLEAN], z=z_internal_trans))

            # y slide
            base_movements.append(
                offset_frame_by_x_y_z(lt + Q + [RobotConfig.SPEED_TRAVEL], z=z_internal_trans))
            base_movements.append(lt + Q + [RobotConfig.SPEED_CLEAN])
            base_movements.append(lb + Q + [RobotConfig.SPEED_CLEAN])
            base_movements.append(
                offset_frame_by_x_y_z(lb + Q + [RobotConfig.SPEED_CLEAN], z=z_internal_trans))

            base_movements.append(save_position)

            transfer_pos_rapid_code = "\t\tMoveL P" + brush_func_name + "_" + str(len(base_movements) - 1) + ", v" \
                                      + str(RobotConfig.SPEED_TRAVEL) + ", fine, pen\\WObj:=paintWO;"

            RapidAPI.make_robtargets_file(movements=base_movements, func_name=brush_func_name)

            with open(os.path.join(paths.rapid_functions_path, brush_func_name + "_Moves.txt"), 'w') as rapid_file:
                writer = RapidAPI(args=self.args, rapid_file=rapid_file)

                writer.write_proc(fn_name=brush_func_name, arg_names_list=["rand0", "rand1", "rand2", "rand3"])

                # Transfer point
                rapid_file.write(transfer_pos_rapid_code)

                # each position in the actual action
                for m in range(4):
                    for frame_idx in range(4):
                        index = (m * 4) + frame_idx
                        if m == 0 or m == 1:
                            x_off_str = "rand" + str(m)
                            y_off_str = "0"
                        else:
                            x_off_str = "0"
                            y_off_str = "rand" + str(m)

                        writer.write_rapid_MoveL_with_offset(pos8=base_movements[index],
                                                             pos_idx=frame_idx,
                                                             point_name=brush_func_name,
                                                             x_off_var_name=x_off_str,
                                                             y_off_var_name=y_off_str,
                                                             z_off_var_name="0",
                                                             fine=False)

                # Safety position transition point
                rapid_file.write(transfer_pos_rapid_code)

                writer.write_end_proc()

    def write_rapid_pump_clean_for_all_brushes(self):

        Q = RobotConfig.STANDARD_QUATERNIONS
        for brush in BrushConfig.ALL_BRUSHES_LIST:
            transfer_position = RobotConfig.POS_WATER_PUMP_TRANSITION_POINT + Q + [
                RobotConfig.SPEED_TRAVEL]
            before_pickup_pos = RobotConfig.POS_WATER_PUMP_TOP + Q + [RobotConfig.SPEED_TRAVEL]
            before_pickup_pos = offset_frame_by_x_y_z(before_pickup_pos, z=brush.delta_z_water)

            water_pos_c = RobotConfig.POS_WATER_PUMP_CLEANING_HEIGHT + Q + [RobotConfig.SPEED_TRAVEL]
            water_pos_l1 = RobotConfig.POS_WATER_PUMP_CLEANING_HEIGHT + Q + [RobotConfig.SPEED_CLEAN]
            water_pos_l2 = RobotConfig.POS_WATER_PUMP_CLEANING_HEIGHT + Q + [RobotConfig.SPEED_CLEAN]
            water_pos_r1 = RobotConfig.POS_WATER_PUMP_CLEANING_HEIGHT + Q + [RobotConfig.SPEED_CLEAN]
            water_pos_r2 = RobotConfig.POS_WATER_PUMP_CLEANING_HEIGHT + Q + [RobotConfig.SPEED_CLEAN]
            water_pos_l1 = offset_frame_by_x_y_z(water_pos_l1, z=brush.delta_z_water)
            water_pos_l2 = offset_frame_by_x_y_z(water_pos_l2, z=brush.delta_z_water)
            water_pos_r1 = offset_frame_by_x_y_z(water_pos_r1, z=brush.delta_z_water)
            water_pos_r2 = offset_frame_by_x_y_z(water_pos_r2, z=brush.delta_z_water)
            water_pos_c = offset_frame_by_x_y_z(water_pos_c, z=brush.delta_z_water)

            n_keyframes = 9

            # DONT CHANGE THIS WITHOUT CHANGING THE INNER-MOST LOOP BELLOW!
            movement_list = [
                transfer_position,
                before_pickup_pos,
                water_pos_c,
                water_pos_r1,
                water_pos_r2,
                water_pos_r1,
                water_pos_c,
                water_pos_l1,
                water_pos_l2,
                water_pos_l1,
                water_pos_c,
                before_pickup_pos,
                transfer_position,
            ]

            func_name = self.args.clean_func_name_prefix + brush.name
            RapidAPI.make_robtargets_file(movements=movement_list, func_name=func_name)

            # function
            with open(os.path.join(paths.rapid_functions_path, func_name + "_Moves.txt"), 'w') as rapid_file:
                writer = RapidAPI(args=self.args, rapid_file=rapid_file)

                writer.write_proc(fn_name=func_name)

                # this is highly dependent on movement structure!
                for i in range(2):
                    writer.write_rapid_MoveL(pos8=movement_list[i], point_name=func_name, pos_idx=i)

                writer.write_start_water()

                for i in range(2, n_keyframes + 2):
                    writer.write_rapid_MoveL(pos8=movement_list[i], point_name=func_name, pos_idx=i)
                    writer.write_rapid_Wait(
                        n_milliseconds=float(RobotConfig.TIME_CLEAN_PUMP_MILLIS) / float(n_keyframes))

                writer.write_stop_water()

                for i in range(len(movement_list) - 2, len(movement_list)):
                    writer.write_rapid_MoveL(pos8=movement_list[i], point_name=func_name, pos_idx=i)

                writer.write_end_proc()

    def write_rapid_all_brush_mount_and_unmount_fns(self):
        Q = RobotConfig.STANDARD_QUATERNIONS
        for brush in BrushConfig.ALL_BRUSHES_LIST:
            transfer_position = RobotConfig.POS_BRUSH_SWAP_TRANSITION_POINT + Q + [RobotConfig.SPEED_TRAVEL]

            # DONT CHANGE THIS WITHOUT CHANGING THE INNER-MOST LOOP BELLOW!
            movement_list = [
                transfer_position,
                brush.position_before_pick_up + Q + [RobotConfig.SPEED_SWAP_BRUSH],
                brush.position_for_pick_up + Q + [RobotConfig.SPEED_SWAP_BRUSH],
                brush.position_before_pick_up + Q + [RobotConfig.SPEED_SWAP_BRUSH],
                transfer_position,
            ]

            # _________________________________________________ MOUNT __________________________________________________
            func_name = self.args.mount_brush_func_name_prefix + brush.name
            RapidAPI.make_robtargets_file(movements=movement_list, func_name=func_name)

            # function
            with open(os.path.join(paths.rapid_functions_path, func_name + "_Moves.txt"), 'w') as rapid_file:
                writer = RapidAPI(args=self.args, rapid_file=rapid_file)

                writer.write_proc(fn_name=func_name)

                # this is HIGHLY dependent on movement structure!
                for i in range(2):
                    writer.write_rapid_MoveL(pos8=movement_list[i], point_name=func_name, pos_idx=i)
                writer.write_open_hydraulic()
                writer.write_rapid_MoveL(pos8=movement_list[2], point_name=func_name, pos_idx=2)
                writer.write_close_hydraulic()
                writer.write_rapid_Wait(n_milliseconds=RobotConfig.WAIT_TIME_MOUNT_UNMOUNT_MILLIS)
                for i in range(3, 5):
                    writer.write_rapid_MoveL(pos8=movement_list[i], point_name=func_name, pos_idx=i)

                writer.write_end_proc()

            # ------------------------------------------------ UNMOUNT -------------------------------------------------

            func_name = self.args.unmount_brush_func_name_prefix + brush.name
            RapidAPI.make_robtargets_file(movements=movement_list, func_name=func_name)

            # function
            with open(os.path.join(paths.rapid_functions_path, func_name + "_Moves.txt"), 'w') as rapid_file:
                writer = RapidAPI(args=self.args, rapid_file=rapid_file)

                writer.write_proc(fn_name=func_name)

                # this is HIGHLY dependent on movement structure!
                for i in range(3):
                    writer.write_rapid_MoveL(pos8=movement_list[i], point_name=func_name, pos_idx=i)
                writer.write_open_hydraulic()
                writer.write_rapid_Wait(n_milliseconds=RobotConfig.WAIT_TIME_MOUNT_UNMOUNT_MILLIS)
                for i in range(3, 5):
                    writer.write_rapid_MoveL(pos8=movement_list[i], point_name=func_name, pos_idx=i)
                writer.write_close_hydraulic()

                writer.write_end_proc()

    def write_rapid_all_brush_swap_fns(self):
        for from_brush in BrushConfig.ALL_BRUSHES_LIST:

            other_brushes = [b for b in BrushConfig.ALL_BRUSHES_LIST if b != from_brush]
            for to_brush in other_brushes:
                func_name = self.args.change_brush_func_name_prefix \
                            + "F_" + from_brush.name + "_TO_" + to_brush.name
                with open(os.path.join(paths.rapid_functions_path, func_name + "_Moves.txt"), 'w') as rapid_file:
                    writer = RapidAPI(args=self.args, rapid_file=rapid_file)

                    writer.write_proc(fn_name=func_name)

                    writer.write_ummount_brush(from_brush=from_brush)
                    writer.write_mount_brush(to_brush=to_brush)

                    writer.write_end_proc()

    def write_rapid_get_color_fns_for_all_brushes(self):
        Q = RobotConfig.STANDARD_QUATERNIONS

        radii_list = []
        for i in range(3):
            n = []
            for j in range(1, 5):
                n.append(RobotConfig.RADIUS_COLOR_CUPS / j - (RobotConfig.RADIUS_COLOR_CUPS / 4 / 3) * i)
                radii_list.append(n)

        for brush in BrushConfig.ALL_BRUSHES_LIST:

            for i in range(len(radii_list)):
                for color in ColorConfig.ALL_COLORS_LIST:
                    func_name = self.args.getcolor_func_name_prefix + color.capital + str(i) + "_" + brush.name
                    movements = []
                    movements += [RobotConfig.POS_COLOR_TRANSITION_POINT + Q + [RobotConfig.SPEED_TRAVEL]]

                    circles = get_n_circles(                                            num_circles=len(radii_list[i]),
                                            max_radius_mm=RobotConfig.RADIUS_COLOR_CUPS,
                                            min_radius_mm=2,
                                            predefined_radii_list=radii_list[i],
                                            center_x=color.position[0],
                                            center_y=color.position[1],
                                            z_mm=color.position[2] + brush.delta_z_color,
                                            v=RobotConfig.SPEED_GET_COLOR)
                    circles = pad_strokes_with_lift_up([circles], z_pad_mm=RobotConfig.COLOR_Z_SAFETY_PAD)[0]
                    circles[0][7] = float(RobotConfig.SPEED_TRAVEL)
                    movements += circles.tolist()
                    movements += [RobotConfig.POS_COLOR_TRANSITION_POINT + Q + [RobotConfig.SPEED_TRAVEL]]

                    RapidAPI.make_robtargets_file(movements=movements, func_name=func_name)

                    with open(os.path.join(paths.rapid_functions_path, func_name + "_Moves.txt"), 'w') as rapid_file:
                        writer = RapidAPI(args=self.args, rapid_file=rapid_file)

                        writer.write_proc(fn_name=func_name, arg_names_list=["z_off"])

                        # safety position
                        writer.write_rapid_MoveL_with_offset(pos8=movements[0],
                                                             point_name=func_name,
                                                             pos_idx=0,
                                                             z_off_var_name="z_off",
                                                             fine=True)

                        # open bucket
                        writer.write_open_bucket(color=color)

                        # each position of the robot movements of getting color
                        for frame_idx in range(1, len(movements) - 1):
                            writer.write_rapid_MoveL_with_offset(pos8=movements[frame_idx],
                                                                 point_name=func_name,
                                                                 pos_idx=frame_idx,
                                                                 z_off_var_name="z_off",
                                                                 fine=False)

                        # safety position
                        writer.write_rapid_MoveL_with_offset(pos8=movements[0],
                                                             point_name=func_name,
                                                             pos_idx=0,
                                                             z_off_var_name="z_off",
                                                             fine=True)

                        # close bucket
                        writer.write_close_bucket(color=color)

                        writer.write_end_proc()

    def _write_stroke_fn(self, stroke_name, stroke_movements):
        RapidAPI.make_robtargets_file(movements=stroke_movements,
                                      func_name=stroke_name)

        with open(os.path.join(paths.rapid_functions_path, stroke_name + "_Moves.txt"), 'w') as rapid_file:
            writer = RapidAPI(args=self.args, rapid_file=rapid_file)
            writer.write_proc(fn_name=stroke_name, arg_names_list=["x_off", "y_off", "z_off"])

            # each position of the robot movements of the stroke
            for idx, pos8 in enumerate(stroke_movements):
                writer.write_rapid_MoveL_with_offset(pos8=pos8,
                                                     point_name=stroke_name,
                                                     pos_idx=idx,
                                                     x_off_var_name="x_off",
                                                     y_off_var_name="y_off",
                                                     z_off_var_name="z_off",
                                                     fine=False)

            writer.write_end_proc()
