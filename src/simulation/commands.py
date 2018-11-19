# Copyright(c) Eric Steinberger 2018

"""
This file contains tools to reorder the strokes that come out of the Iterative Stroke Sampling simulation.
We optimize for swapping color and brush as rarely as possible, by reordering strokes that wouldn't overpaint eachother.
"""

import copy
import os

import numpy as np

from src import paths
from src.config import BrushConfig
from src.rapid.RapidFileWriter import RapidFileWriter


# ________________________________________________ SINGLE COMMANDS _____________________________________________________

class Command(object):
    def get_rapid(self):
        raise NotImplementedError

    def is_get_color(self):
        return isinstance(self, GetColor)

    def is_clean(self):
        return isinstance(self, Clean)

    def is_change_brush(self):
        return isinstance(self, ChangeBrush)

    def is_apply_stroke(self):
        return isinstance(self, ApplyStroke)


class GetColor(Command):
    def __init__(self, color):
        self.color = color

    def get_rapid(self):
        return "get" + self.color.capital


class Clean(Command):
    def __init__(self):
        pass

    def get_rapid(self):
        return "Clean"


class ChangeBrush(Command):
    def __init__(self, from_brush, to_brush):
        self.from_brush = from_brush
        self.to_brush = to_brush

    def get_rapid(self):
        return "ChBr_F_" + self.from_brush.name + "_TO_" + self.to_brush.name


class ApplyStroke(Command):
    def __init__(self, brush, color, stroke_id, rotation_id,
                 center_x_mm, center_y_mm, painting_size_x_mm, painting_size_y_mm, canvas_center_xyz_mm):
        self.brush = brush
        self.color = color  # just stored for info in order optimizer
        self.stroke_id = stroke_id
        self.rotation_id = rotation_id
        self.x = canvas_center_xyz_mm[0] + center_x_mm - (painting_size_x_mm / 2.0)
        self.y = canvas_center_xyz_mm[1] + center_y_mm - (painting_size_y_mm / 2.0)
        self.z = canvas_center_xyz_mm[2]

    def get_rapid(self):
        func_name = self.brush.stroke_names_list[self.stroke_id] + "_" + str(self.rotation_id + 1)
        return func_name + " " + str(self.x) + ", " + str(self.y) + ", " + str(self.z)


# ______________________________________________ SEQUENCE OF COMMANDS __________________________________________________
class CommandSequence(object):

    def __init__(self, args):
        self.args = args
        self.name = args.name
        self.commands = None
        self.reset()

    def append(self, x):
        self.commands.append(x)

    def optimize_order(self):
        print("Cleaning command sequence")
        self.commands = _clean_list_of_commands(list_of_commands=self.commands)

        print("Optimizing command order for least amount of color & brush swaps.")
        self.commands = StrokeOrderOptimizer.run(list_of_commands=self.commands, args=self.args)

        # Do again after order optimization
        self.commands = _clean_list_of_commands(list_of_commands=self.commands)

    def write_rapid_code_to_file(self):
        with open(os.path.join(paths.paintings_path, self.name + ".txt"), 'w') as rapid_file:
            rapid_file_writer = RapidFileWriter(rapid_file, args=self.args)

            rapid_file_writer.begin_module()
            rapid_file_writer.import_robtargets_from_all_rapid_fns()

            # __________________________________________________ main __________________________________________________
            rapid_file.write("\n\n\t" + "PROC main()\n")
            rapid_file_writer.call_activate_feedback_pin()

            # just to track for print statement
            num_cleaned = 0
            num_changed_brush = 0

            current_brush = None

            for cmd in self.commands:

                if cmd.is_clean():
                    num_cleaned += 1
                    rapid_file_writer.call_pump_clean(brush=current_brush)
                    rapid_file_writer.call_towel(brush=current_brush)
                    rapid_file_writer.call_pump_clean(brush=current_brush)
                    rapid_file_writer.call_towel(brush=current_brush)

                elif cmd.is_get_color():
                    rapid_file_writer.call_get_color(brush=current_brush, color=cmd.color)

                elif cmd.is_change_brush():
                    num_changed_brush += 1
                    current_brush = cmd.to_brush
                    rapid_file_writer.call_brush_swap(from_brush=cmd.from_brush, to_brush=cmd.to_brush)

                elif cmd.is_apply_stroke():
                    rapid_file.write("\t\t" + cmd.get_rapid() + ";" + "\n")

                else:
                    raise ValueError(cmd)

            assert current_brush.name == BrushConfig.NOTHING_MOUNTED.name

            rapid_file_writer.end_proc()
            rapid_file_writer.import_all_function_definitions()
            rapid_file_writer.end_module()

        print("Changed brush ", num_changed_brush, " times.")
        print("Cleaned ", num_cleaned, " times.")

    def reset(self):
        self.commands = []


# ____________________________________________ TOOLS FOR SEQ OF COMMANDS _______________________________________________
def _clean_list_of_commands(list_of_commands):
    """ scans for and deletes redundant stuff in code like get_color -> Clean """
    _list_of_commands = copy.copy(list_of_commands)

    found_smth = True
    while found_smth:
        i = 0
        curr_len = len(_list_of_commands)
        found_smth = False
        while i < curr_len - 1:

            # get_color -> Clean
            if _list_of_commands[i].is_get_color():
                if _list_of_commands[i + 1].is_clean():
                    del _list_of_commands[i]  # delete get_color
                    found_smth = True

            # get_color -> Change_Brush
            elif _list_of_commands[i].is_get_color():
                if _list_of_commands[i + 1].is_change_brush():
                    del _list_of_commands[i]  # delete get_color
                    found_smth = True

            # Change_Brush -> Clean
            elif _list_of_commands[i].is_change_brush():
                if _list_of_commands[i + 1].is_clean():
                    del _list_of_commands[i + 1]  # delete Clean
                    found_smth = True

            # Change_Brush -> Change_Brush
            elif _list_of_commands[i].is_change_brush():
                if _list_of_commands[i + 1].is_change_brush():
                    del _list_of_commands[i]  # delete first Change_Brush
                    found_smth = True

            # Change_Brush to same brush
            elif _list_of_commands[i].is_change_brush():
                if _list_of_commands[i].from_brush is _list_of_commands[i].to_brush:
                    del _list_of_commands[i]  # delete Change_Brush
                    found_smth = True

            # get_color -> get_color
            elif _list_of_commands[i].is_get_color():
                if _list_of_commands[i + 1].is_get_color():
                    del _list_of_commands[i]  # delete first get_color
                    found_smth = True

            # Clean -> Clean
            elif _list_of_commands[i].is_clean():
                if _list_of_commands[i + 1].is_clean():
                    del _list_of_commands[i]  # delete first Clean
                    found_smth = True

            i += 1
            curr_len = len(_list_of_commands)

    return _list_of_commands


class _OptimizerStrokeBlock:
    def __init__(self, color, brush):
        self.strokes_list = []
        self.brush = brush
        self.color = color


class StrokeOrderOptimizer:
    @staticmethod
    def run(list_of_commands, args):
        _list_of_commands = copy.copy(list_of_commands)

        list_of_strokes = StrokeOrderOptimizer._build_strokes_list(list_of_commands=_list_of_commands)
        stroke_blocks_list = StrokeOrderOptimizer._build_stroke_blocks(list_of_strokes=list_of_strokes)

        # optimize order
        did_something = True
        while did_something:
            did_something = StrokeOrderOptimizer._optimize(stroke_blocks_list=stroke_blocks_list,
                                                           stroke_size=args.stroke_size_mm)

        # return list of cmds like we got it, but with optimized order
        return StrokeOrderOptimizer._list_of_blocks_to_list_of_commands(stroke_blocks_list)

    @staticmethod
    def _list_of_blocks_to_list_of_commands(stroke_blocks_list,):
        commands = []
        last_brush = BrushConfig.BRUSH_AT_BOOT
        for block in stroke_blocks_list:

            # swap brush if not equal to old one
            if last_brush is not block.brush:
                if last_brush is not BrushConfig.NOTHING_MOUNTED:
                    commands.append(Clean())
                commands.append(ChangeBrush(from_brush=last_brush, to_brush=block.brush))
                last_brush = block.brush

            for stroke in block.strokes_list:
                # after every stroke get color
                commands.append(GetColor(color=stroke.color))
                commands.append(stroke)

            # Clean brush at end of every block
            commands.append(Clean())

        commands.append(ChangeBrush(from_brush=last_brush, to_brush=BrushConfig.NOTHING_MOUNTED))
        return commands

    @staticmethod
    def _overlaps(block_1, block_2, stroke_size):
        # loop over all strokes in ith block
        for stroke_idx_loop in range(len(block_1.strokes_list)):

            # loop over all strokes in 2nd merge_block
            for stroke_idx_2nd in range(len(block_2.strokes_list)):

                # check if strokes have spacial overlap
                x_distance = np.abs(block_2.strokes_list[stroke_idx_2nd].x - block_1.strokes_list[stroke_idx_loop].x)
                y_distance = np.abs(block_2.strokes_list[stroke_idx_2nd].y - block_1.strokes_list[stroke_idx_loop].y)

                eucl = np.sqrt(x_distance ** 2 + y_distance ** 2)

                if eucl < stroke_size:
                    return True

        return False

    @staticmethod
    def _optimize_front_to_back(stroke_blocks_list, stroke_size):
        did_something = False
        block_idx = 0

        while block_idx < len(stroke_blocks_list):
            color = stroke_blocks_list[block_idx].color
            brush = stroke_blocks_list[block_idx].brush

            # search next color group with that same color and brush
            idx_next_block_using_color_and_brush = -1

            for block_idx2 in range(block_idx + 1, len(stroke_blocks_list)):
                if color == stroke_blocks_list[block_idx2].color and brush == stroke_blocks_list[block_idx2].brush:
                    idx_next_block_using_color_and_brush = block_idx2
                    break

            # check if color and brush are used ever again. If there is one, check if can merge
            if idx_next_block_using_color_and_brush != -1:
                can_merge = True

                # loop over all blocks between merge_block 1 and 2 that we maybe want to merge
                for loop_block_idx in range(block_idx + 1, idx_next_block_using_color_and_brush):

                    if StrokeOrderOptimizer._overlaps(block_1=stroke_blocks_list[idx_next_block_using_color_and_brush],
                                                      block_2=stroke_blocks_list[loop_block_idx],
                                                      stroke_size=stroke_size):
                        can_merge = False
                        break

                if can_merge is True:
                    copied_list = copy.copy(stroke_blocks_list[idx_next_block_using_color_and_brush].strokes_list)
                    stroke_blocks_list[block_idx].strokes_list += copied_list
                    del stroke_blocks_list[idx_next_block_using_color_and_brush]
                    did_something = True

            block_idx += 1

        return did_something

    @staticmethod
    def _optimize_back_to_front(stroke_blocks_list, stroke_size):
        did_something = False
        block_idx = 0
        while block_idx < len(stroke_blocks_list):
            did_something_now = False

            color = stroke_blocks_list[block_idx].color
            brush = stroke_blocks_list[block_idx].brush

            # search next color group with that same color and brush
            idx_next_block_using_color_and_brush = -1

            for block_idx2 in reversed(range(0, block_idx)):
                if color == stroke_blocks_list[block_idx2].color and brush == stroke_blocks_list[block_idx2].brush:
                    idx_next_block_using_color_and_brush = block_idx2
                    break

            # check if color and brush are used ever again. If there is one, check if can merge
            if idx_next_block_using_color_and_brush != -1:
                can_merge = True

                # loop over all blocks between merge_block 1 and 2 that we maybe want to merge
                for loop_block_idx in range(idx_next_block_using_color_and_brush + 1, block_idx):

                    if StrokeOrderOptimizer._overlaps(
                            block_1=stroke_blocks_list[idx_next_block_using_color_and_brush],
                            block_2=stroke_blocks_list[loop_block_idx],
                            stroke_size=stroke_size):
                        can_merge = False
                        break

                if can_merge is True:
                    copied_list = copy.copy(stroke_blocks_list[block_idx].strokes_list)
                    stroke_blocks_list[idx_next_block_using_color_and_brush].strokes_list += copied_list
                    del stroke_blocks_list[block_idx]
                    did_something = True
                    did_something_now = True

            if not did_something_now:  # else we deleted one and would skip one if we did += 1.
                block_idx += 1

        return did_something

    @staticmethod
    def _optimize(stroke_blocks_list, stroke_size):
        assert isinstance(stroke_blocks_list, list)
        for s in stroke_blocks_list:
            assert isinstance(s, _OptimizerStrokeBlock)

        # runs optimization and returns boolean whether anything was changed
        return (StrokeOrderOptimizer._optimize_front_to_back(stroke_blocks_list=stroke_blocks_list,
                                                             stroke_size=stroke_size)
                or StrokeOrderOptimizer._optimize_back_to_front(stroke_blocks_list=stroke_blocks_list,
                                                                stroke_size=stroke_size))

    @staticmethod
    def _build_strokes_list(list_of_commands):
        list_of_strokes = []
        current_color = None
        current_brush = None

        for cmd in list_of_commands:

            if cmd.is_get_color():
                current_color = cmd.color

            elif cmd.is_change_brush():
                current_brush = cmd.to_brush

            elif cmd.is_apply_stroke():
                assert cmd.brush is current_brush
                assert cmd.color is current_color
                list_of_strokes.append(cmd)

        return list_of_strokes

    @staticmethod
    def _build_stroke_blocks(list_of_strokes):
        assert isinstance(list_of_strokes, list)
        for s in list_of_strokes:
            assert isinstance(s, ApplyStroke)

        list_of_stroke_blocks = []
        current_color = list_of_strokes[0].color
        current_brush = list_of_strokes[0].brush

        stroke_block = _OptimizerStrokeBlock(color=current_color, brush=current_brush)
        list_of_stroke_blocks.append(stroke_block)
        for stroke in list_of_strokes:
            if stroke.color == current_color and stroke.brush == current_brush:
                stroke_block.strokes_list.append(stroke)

            else:
                # reset and start new block
                current_color = stroke.color
                current_brush = stroke.brush
                stroke_block = _OptimizerStrokeBlock(color=current_color, brush=current_brush)
                list_of_stroke_blocks.append(stroke_block)

                # append this stroke
                stroke_block.strokes_list.append(stroke)

        return list_of_stroke_blocks
