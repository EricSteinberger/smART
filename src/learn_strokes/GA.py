# Copyright(c) Eric Steinberger 2018


"""
To learn strokes smART uses a very simple Genetic Algorithm without crossover with manual selection (i.e. user input).
"""

import copy
import os

import numpy as np

from src import paths
from src.Brush import Brush
from src.config import RobotConfig, ColorConfig
from src.rapid.RapidFileWriter import RapidFileWriter
from src.rapid.RapidFunctionWriter import RapidFunctionWriter
from src.simulation.stroke_tools import get_base_pencil_stroke_from_a_to_b, offset_stroke_by_x_y_z, \
    save_GA_stroke_as_npy
from src.simulation.stroke_tools import pad_strokes_with_lift_up, flatten_3d_list_of_strokes_lists_to_2d_position_list


class GA:

    def __init__(self,
                 args,
                 brush_to_paint_with,
                 brush_currently_on,
                 stroke_name,
                 n_different_strokes_per_generation=12,
                 start_stroke_length_mm=28,
                 stroke_deepness_mm=1.5,
                 how_often_paint_each_stroke=5,
                 color=ColorConfig.BLACK,
                 build_fns=False):

        self.writer = RapidFunctionWriter(args=args)
        if build_fns:
            self.writer.write_all()

        assert isinstance(brush_to_paint_with, Brush)
        self.args = args
        self.stroke_name = stroke_name
        self.brush = brush_to_paint_with
        self.brush_currently_on = brush_currently_on
        self.color = color
        self.num_different_strokes_per_generation = n_different_strokes_per_generation
        self.how_often_paint_each_stroke = how_often_paint_each_stroke
        self.start_stroke_length = start_stroke_length_mm
        self.stroke_deepness_mm = stroke_deepness_mm
        self.curr_strokes = [get_base_pencil_stroke_from_a_to_b(start_x_mm=0,
                                                                start_y_mm=0,
                                                                z_paper_mm=RobotConfig.POS_PAPER_A4_PAINT_LB[
                                                                               2] + self.brush.delta_z_paint,
                                                                stroke_length_mm=self.start_stroke_length,
                                                                number_of_points_between_start_and_end=10,
                                                                stroke_deepness_mm=self.stroke_deepness_mm,
                                                                v=100,
                                                                safety_pad_mm=0)

                             for _ in range(self.num_different_strokes_per_generation)]
        self.gens_done = 0
        self.list_of_best_two_stroke_indices = [0, 1]

    def next_generation(self):
        self.next_pop_from_2_best(list_of_best_2_stroke_indices=self.list_of_best_two_stroke_indices)

        self.draw_next_generation()

        # export current generation to disk for later usage in the stroke library
        for i in range(len(self.curr_strokes)):
            save_GA_stroke_as_npy(one_stroke_array=np.array(self.curr_strokes[i]),
                                  name="gen_" + str(self.gens_done) + "_stroke_" + str(i),
                                  brush_name=self.brush.name,
                                  stroke_name=self.stroke_name)

        self.list_of_best_two_stroke_indices = self.select_best_2()

        self.gens_done += 1

    def select_best_2(self):
        def _get_int_input(text):
            while True:
                try:
                    o = int(input(text))
                except ValueError:
                    print("Invalid input! Please try again.")
                    continue
                break
            return o

        best1 = _get_int_input("enter number of best stroke: ")
        best2 = _get_int_input("enter number of 2nd best stroke: ")
        return [best1, best2]

    def draw_next_generation(self):
        strokes_to_paint = []
        x_off = 20
        y_off = 18
        x_start = RobotConfig.POS_PAPER_A4_PAINT_LB[0]
        y_start = RobotConfig.POS_PAPER_A4_PAINT_LB[1]

        # pad with safe-room on Z axis on start and end frames
        padded_strokes = pad_strokes_with_lift_up(self.curr_strokes, z_pad_mm=50)
        padded_strokes = np.array(padded_strokes)
        for s in range(len(padded_strokes)):

            for i in range(self.how_often_paint_each_stroke):
                off_stroke = offset_stroke_by_x_y_z(stroke=padded_strokes[s], x=x_start - (i * x_off),
                                                    y=y_start - (s * y_off))
                strokes_to_paint.append(off_stroke)

        self.write_rapid_movements_for_ga(strokes=strokes_to_paint,
                                          rapid_file_name=str(self.gens_done) + "_strokes",
                                          from_brush=self.brush_currently_on,
                                          to_brush=self.brush,
                                          clean_freq=self.how_often_paint_each_stroke)
        self.brush_currently_on = self.brush

    def next_pop_from_2_best(self, list_of_best_2_stroke_indices):
        list_of_best_2_strokes = [self.curr_strokes[list_of_best_2_stroke_indices[0]],
                                  self.curr_strokes[list_of_best_2_stroke_indices[1]]]

        self.curr_strokes = []

        n_first = int(self.num_different_strokes_per_generation / 2)
        n_second = self.num_different_strokes_per_generation - n_first

        for x in range(n_first):
            self.curr_strokes.append(list_of_best_2_strokes[0])
        for x in range(n_second):
            self.curr_strokes.append(list_of_best_2_strokes[1])

        delta_movements = self.calculate_delta_movements()
        delta_movements = self.mutate(delta_movements)
        self.calculate_strokes_from_delta_movements(delta_movements)

    def mutate(self, delta_movements, z_min=-4):

        _delta_movements = delta_movements[:]

        def get_mutation_for_x_or_y():
            return np.random.normal()

        def get_mutation_for_z(current_z, multiplier=0.05):
            off = get_mutation_for_x_or_y() * multiplier
            if off + current_z < z_min:
                return 0.0, current_z
            return off, (current_z + off)

        # ___________________________________ make_random_changes_to_delta_movements ___________________________________
        for s in range(len(_delta_movements)):
            curr_z = self.curr_strokes[s][0][2]

            for d in range(len(_delta_movements[s])):
                _delta_movements[s][d][0] += get_mutation_for_x_or_y()
                _delta_movements[s][d][1] += get_mutation_for_x_or_y()
                curr_z += _delta_movements[s][d][2]
                z_off, curr_z = get_mutation_for_z(curr_z)
                _delta_movements[s][d][2] += z_off
        return _delta_movements

    def calculate_delta_movements(self):
        delta_movements = []
        for x in range(self.num_different_strokes_per_generation):
            stroke = self.curr_strokes[x]
            delta_stroke = []

            for p in range(1, len(stroke)):
                sss = []
                for i in range(7):
                    sss.append(stroke[p][i] - stroke[p - 1][i])
                delta_stroke.append(sss)

            delta_movements.append(delta_stroke)
        return delta_movements

    def calculate_strokes_from_delta_movements(self, delta_movements):
        strokes = []
        for x in range(self.num_different_strokes_per_generation):
            stroke = copy.deepcopy(self.curr_strokes[x])
            for p in range(1, len(stroke)):
                for i in range(7):
                    stroke[p][i] = stroke[p - 1][i] + delta_movements[x][p - 1][i]

            strokes.append(stroke)

        self.curr_strokes = strokes

    def write_rapid_movements_for_ga(self, strokes, clean_freq, rapid_file_name, from_brush, to_brush,
                                     color=None):
        if color is None:
            color = ColorConfig.BLACK

        with open(os.path.join(paths.rapid_GA_path, "RAPID_" + rapid_file_name + ".txt"), 'w') as rapid_file:
            rapid_file_writer = RapidFileWriter(rapid_file=rapid_file, args=self.args)
            rapid_file_writer.begin_module()

            rapid_file_writer.import_robtargets_from_all_rapid_fns()

            # robtargets for the experimental strokes
            all_pos = flatten_3d_list_of_strokes_lists_to_2d_position_list(strokes)
            for frame in range(len(all_pos)):
                iter_str = str(frame)
                px = float(all_pos[frame][0])
                py = float(all_pos[frame][1])
                pz = float(all_pos[frame][2])
                rw = float(all_pos[frame][3])
                rx = float(all_pos[frame][4])
                ry = float(all_pos[frame][5])
                rz = float(all_pos[frame][6])
                px = str("{0:.7f}".format(round(px, 7)))
                py = str("{0:.7f}".format(round(py, 7)))
                pz = str("{0:.7f}".format(round(pz, 7)))
                rw = str("{0:.7f}".format(round(rw, 7)))
                rx = str("{0:.7f}".format(round(rx, 7)))
                ry = str("{0:.7f}".format(round(ry, 7)))
                rz = str("{0:.7f}".format(round(rz, 7)))
                loc = "\t CONST robtarget PDraw_" + iter_str + ":=[[" + px + "," + py + "," + pz \
                      + "],[" + rw + "," + rx + "," + ry + "," + rz \
                      + "],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];"
                rapid_file.write(loc + "\n")

            rapid_file.write("\n\n\t PROC main()\n")
            rapid_file_writer.call_activate_feedback_pin()

            # maybe swap brush
            if from_brush is not to_brush:
                assert from_brush is not None
                assert to_brush is not None
                rapid_file_writer.call_brush_swap(from_brush=from_brush, to_brush=to_brush)

            # movements for the experimental strokes
            pos_counter = 0
            n_done = 0
            for stroke in strokes:
                rapid_file_writer.call_get_color(brush=to_brush, color=color)
                for frame in stroke:
                    rapid_file_writer.rapid_api.write_rapid_MoveL(pos8=frame, point_name="Draw", pos_idx=pos_counter)
                    pos_counter += 1

                n_done += 1

                if n_done % clean_freq == 0 and clean_freq != 0:
                    rapid_file_writer.call_pump_clean(brush=to_brush)
                    rapid_file_writer.call_towel(brush=to_brush)

            rapid_file_writer.end_proc()
            rapid_file_writer.import_all_function_definitions()
            rapid_file_writer.end_module()
