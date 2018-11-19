# Copyright(c) Eric Steinberger 2018

"""

[0,0] is at X

    #  _________________________
    # |                         |
    # |                         |
    # |                         |
    # |                         |
    # |                         |
    # |                         |
    #  ____________X____________

    #         ROBOT HERE


All positions are defined relative to the point [0, 0, 0], where the last axis (Z) is the vertical one.
Values on the Z axis are offset by a non-zero constant that emerges from how our physical setup is built.

This file contains the necessary measured positions and other attributes of all objects smART interacts with.
"""

# _______________________________________________ POSITIONS ____________________________________________________
# -- default
REFERENCE_BRUSH_Z_MEASSURE = 120.3  # P # WORLD WHEN TOUCHING GROUND
RS_TOOL_BRUSH_Z = 190  # P # WORLD WHEN TOUCHING GROUND
HYDRAULIC_plus_METAL_HEIGHT_Z = 65.35
STANDARD_QUATERNIONS = [0.00217, 0.34764, 0.93763, 0.0]  # 0°
CLEAN_QUATERNIONS_L1 = [0.00185, -0.91236, -0.40893, 0.0]  # 90°
CLEAN_QUATERNIONS_L2 = [0.00120, -0.94212, 0.33528, 0.00142]  # 180°
CLEAN_QUATERNIONS_R1 = [0.0, 0.41474, -0.90994, -0.00185]  # -90°
CLEAN_QUATERNIONS_R2 = [0.00146, -0.93023, 0.366697, 0.00146]  # -180°

UCS = [242.9, 0, REFERENCE_BRUSH_Z_MEASSURE - RS_TOOL_BRUSH_Z]

# - Trasition Points
POS_BRUSH_SWAP_TRANSITION_POINT = [278.1 - UCS[0], -310.0 - UCS[1], 80 - UCS[2]]
POS_WATER_PUMP_TRANSITION_POINT = POS_BRUSH_SWAP_TRANSITION_POINT
POS_TOWEL_TRANSITION_POINT = POS_BRUSH_SWAP_TRANSITION_POINT
POS_COLOR_TRANSITION_POINT = [278.1 - UCS[0], 200.0 - UCS[1], 60 - UCS[2]]

# -- Safety & Padding
PAINT_SAFETY_PADDING_Z = 50
PAINT_PADD_SIDES = 50
COLOR_Z_SAFETY_PAD = 80
WATER_PUMP_Z_SAFETY_PAD = 70

# -- Object Information
POS_PAPER_A4_PAINT_LB = [UCS[0] - 210 / 2 + PAINT_PADD_SIDES, UCS[1] + 297.0 / 2 - PAINT_PADD_SIDES, -39.0 - UCS[2]]
POS_PAPER_A4_PAINT_LT = [UCS[0] + 210 / 2 - PAINT_PADD_SIDES, UCS[1] + 297.0 / 2 - PAINT_PADD_SIDES, -39.0 - UCS[2]]
POS_PAPER_A4_PAINT_RB = [UCS[0] - 210 / 2 + PAINT_PADD_SIDES, UCS[1] - 297.0 / 2 + PAINT_PADD_SIDES, -39.0 - UCS[2]]
POS_PAPER_A4_PAINT_RT = [UCS[0] + 210 / 2 - PAINT_PADD_SIDES, UCS[1] - 297.0 / 2 + PAINT_PADD_SIDES, -39.0 - UCS[2]]

A3_OFFSET_X = (210.0 - 297.0) / 2.0

RADIUS_COLOR_CUPS = 22.5 - 6  # actual is 22.5  # 6 is safety!
CUP_INNER_HEIGHT = 33.5
_c_inner_height_low = -23.9 - CUP_INNER_HEIGHT
_c_inner_height_high = -13.4 - CUP_INNER_HEIGHT
POS_COLOR_GREEN = [-152.6 - UCS[0], 312.30 - UCS[1], _c_inner_height_low - UCS[2]]
POS_COLOR_YELLOW = [-152.2 - UCS[0], 405.2 - UCS[1], _c_inner_height_low - UCS[2]]
POS_COLOR_BLACK = [162 - UCS[0], 305.7 - UCS[1], _c_inner_height_low - UCS[2]]
POS_COLOR_BROWN = [162 - UCS[0], 403 - UCS[1], _c_inner_height_low - UCS[2]]
POS_COLOR_BLUE = [50.6 - UCS[0], 402.5 - UCS[1], _c_inner_height_high - UCS[2]]
POS_COLOR_RED = [49.9 - UCS[0], 309.4 - UCS[1], _c_inner_height_high - UCS[2]]
POS_COLOR_WHITE = [-42.2 - UCS[0], 405.2 - UCS[1], _c_inner_height_high - UCS[2]]
POS_COLOR_ORANGE = [-43.5 - UCS[0], 309.4 - UCS[1], _c_inner_height_high - UCS[2]]

POS_BRUSH_STATION_1 = [-143.2 - UCS[0], -404 - UCS[1], -70.8]
POS_BRUSH_STATION_2 = [-86.5 - UCS[0], -404.5 - UCS[1], -70.8]
POS_BRUSH_STATION_3 = [-28.6 - UCS[0], -404.9 - UCS[1], -70.8]
POS_BRUSH_NOTHING_MOUNTED = [40 - UCS[0], -404 - UCS[1], 5.0]

POS_WATER_PUMP_CLEANING_HEIGHT = [198.5 - UCS[0], -406.8 - UCS[1], -15.8 - UCS[2]]
POS_WATER_PUMP_TOP = [198.5 - UCS[0], -406.8 - UCS[1], 40.8 + WATER_PUMP_Z_SAFETY_PAD - UCS[2]]

POS_TOWEL_CornerLT = [7.8 - UCS[0], -195.7 - UCS[1], -20.5 - UCS[2]]
POS_TOWEL_CornerLB = [-167.3 - UCS[0], -195.7 - UCS[1], -20.5 - UCS[2]]
POS_TOWEL_CornerRT = [7.8 - UCS[0], -266.3 - UCS[1], -20.5 - UCS[2]]
POS_TOWEL_CornerRB = [-167.3 - UCS[0], -266.3 - UCS[1], -20.5 - UCS[2]]

CANVAS_CENTER = [115.0, 0, 0]

# _________________________________________________ SPEED ______________________________________________________
SPEED_TRAVEL = 400
SPEED_CLEAN = 300
SPEED_STROKE = 200
SPEED_GET_COLOR = 300
SPEED_SWAP_BRUSH = 300
SPEED_WATER = 200

WAIT_TIME_MOUNT_UNMOUNT_MILLIS = 0.8
TIME_CLEAN_PUMP_MILLIS = 7

# __________________________________________________ PINS ______________________________________________________
out_pins = [10, 12, 13, 14]
feedback_pin_o = 11
feedback_pin_i = 14
hydraulic_io_pin = 4
