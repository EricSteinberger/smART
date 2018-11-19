# Copyright(c) Eric Steinberger 2018

"""
Utilities to modify and create brush movements for drawing, cleaning, etc.
"""

import copy
import math
import os

import numpy as np

from src import paths
from src.config import RobotConfig
from src.util import file_util


def pad_strokes_with_lift_up(list_of_strokes, z_pad_mm=15.0):
    padded_list_of_strokes = copy.deepcopy(list_of_strokes)
    # adds lift up in the beginning and end of stroke_sim
    for i in range(len(padded_list_of_strokes)):
        padded_list_of_strokes[i] = np.insert(arr=padded_list_of_strokes[i], obj=0,
                                              values=np.copy(padded_list_of_strokes[i][0]), axis=0)
        padded_list_of_strokes[i][0, 2] += z_pad_mm
        padded_list_of_strokes[i] = np.insert(arr=padded_list_of_strokes[i], obj=padded_list_of_strokes[i].shape[0],
                                              values=np.copy(padded_list_of_strokes[i][-1]), axis=0)
        padded_list_of_strokes[i][-1, 2] += z_pad_mm

    return padded_list_of_strokes


def get_base_pencil_stroke_from_a_to_b(start_x_mm,
                                       start_y_mm,
                                       z_paper_mm,
                                       stroke_length_mm,
                                       v,
                                       stroke_deepness_mm=3,
                                       safety_pad_mm=50,
                                       number_of_points_between_start_and_end=10):
    Q = RobotConfig.STANDARD_QUATERNIONS
    movements = [
        [start_x_mm, start_y_mm, z_paper_mm + safety_pad_mm] + Q + [int(v)],
        [start_x_mm, start_y_mm, z_paper_mm] + Q + [int(v)]
    ]

    distance_per_step = stroke_length_mm / number_of_points_between_start_and_end

    for i in range(number_of_points_between_start_and_end):
        x = start_x_mm + distance_per_step * i
        z = z_paper_mm - stroke_deepness_mm
        movements.append([x, start_y_mm, z] + Q + [int(v)])

    end_x_mm = start_x_mm + stroke_length_mm
    movements.append([end_x_mm, start_y_mm, z_paper_mm] + Q + [int(v)])
    movements.append([end_x_mm, start_y_mm, z_paper_mm + safety_pad_mm] + Q + [int(v)])

    return movements


def get_n_circles(num_circles,
                  center_x,
                  center_y,
                  z_mm,
                  max_radius_mm,
                  min_radius_mm,
                  predefined_radii_list=None,
                  safety_pad_mm=50,
                  v=100,
                  n_increments=18):
    Q = RobotConfig.STANDARD_QUATERNIONS

    movement_list = [
        [center_x, center_y, z_mm + safety_pad_mm] + Q + [int(v)]
    ]

    for i in range(num_circles):

        if predefined_radii_list is None:
            radius = min_radius_mm + (max_radius_mm - min_radius_mm) * np.random.random()
        else:
            radius = predefined_radii_list[i]

        for x in range(n_increments):
            x, y = point_on_circle(angle_in_degrees=(360 / n_increments) * x,
                                   radius=radius,
                                   center_x=center_x,
                                   center_y=center_y)

            movement_list.append([x, y, z_mm] + Q + [int(v)])

    movement_list.append([center_x, center_y, z_mm + safety_pad_mm] + Q + [int(v)])
    return movement_list


def create_rotated_versions_of_stroke(stroke, num_rotations):
    list_of_rotated_strokes = []

    for i in range(num_rotations):
        tmp_stroke = np.copy(stroke)
        angle_in_degrees = i * (360 / num_rotations)

        for frame in range(len(tmp_stroke)):
            rotated_xy = rotate_point([tmp_stroke[frame, 0], stroke[frame, 1]], angle_in_degrees=angle_in_degrees)

            tmp_stroke[frame, 0] = rotated_xy[0]
            tmp_stroke[frame, 1] = rotated_xy[1]

        list_of_rotated_strokes.append(tmp_stroke)

    return list_of_rotated_strokes


def save_GA_stroke_as_npy(one_stroke_array, brush_name, stroke_name, name):
    path = os.path.join(paths.stroke_movements_npy_path, str(stroke_name))

    if not os.path.isdir(path):
        os.makedirs(path)

    path = os.path.join(paths.GA_stroke_npy_path, str(brush_name), str(stroke_name))
    file_util.create_dir_if_not_exist(path)
    file_path = os.path.join(path, str(name) + ".npy")

    np.save(file=file_path, arr=one_stroke_array)


def offset_stroke_by_x_y_z(stroke, x=0.0, y=0.0, z=0.0):
    off_stroke = np.copy(stroke)
    for f in range(len(off_stroke)):
        off_stroke[f] = offset_frame_by_x_y_z(off_stroke[f], x=x, y=y, z=z)
    return off_stroke


def offset_frame_by_x_y_z(frame, x=0.0, y=0.0, z=0.0):
    off_frame = np.copy(frame)
    off_frame[0] += x
    off_frame[1] += y
    off_frame[2] += z
    return off_frame


def flatten_3d_list_of_strokes_lists_to_2d_position_list(list_of_stroke_lists):
    master_list = []
    for a in list_of_stroke_lists:
        for b in a:
            master_list.append(b)

    return master_list


def get_centered_stroke(movements):
    """ Returns a copy of the parameter movements that is centered so that the positively and the negatively
     most distant positions on the X and Y axis are equally far from the center. """

    _stroke = np.copy(movements)

    max_x = _stroke[0, 0]
    max_y = _stroke[0, 1]
    min_x = _stroke[0, 0]
    min_y = _stroke[0, 1]

    for k in range(get_arr_or_list_len(movements=movements)):
        x = _stroke[k, 0]
        y = _stroke[k, 1]
        if x > max_x:
            max_x = x
        elif x < min_x:
            min_x = x
        if y > max_y:
            max_y = y
        elif y < min_y:
            min_y = y

    x_off = -(max_x + min_x) / 2
    y_off = -(max_y + min_y) / 2

    _stroke = offset_stroke_by_x_y_z(_stroke, x_off, y_off)
    return _stroke


def point_on_circle(angle_in_degrees, radius, center_x, center_y):
    from math import cos, sin

    angle = np.radians(angle_in_degrees)
    x = center_x + (radius * cos(angle))
    y = center_y + (radius * sin(angle))

    return x, y


def rotate_point(point, angle_in_degrees, center_point=(0, 0)):
    """Rotates style_ph point around another center_point. Angle is in degrees.
    Rotation is counter-clockwise"""

    angle = math.radians(angle_in_degrees)
    temp_point = point[0] - center_point[0], point[1] - center_point[1]
    temp_point = (temp_point[0] * math.cos(angle) - temp_point[1] * math.sin(angle),
                  temp_point[0] * math.sin(angle) + temp_point[1] * math.cos(angle))
    temp_point = temp_point[0] + center_point[0], temp_point[1] + center_point[1]
    return temp_point


def get_arr_or_list_len(movements):
    if isinstance(movements, np.ndarray):
        return movements.shape[0]
    elif isinstance(movements, list):
        return len(movements)
    else:
        raise ValueError(movements)
