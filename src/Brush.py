# Copyright(c) Eric Steinberger 2018


import copy


class Brush:
    def __init__(self,
                 name,
                 height,
                 reference_height,
                 delta_z_paint,
                 delta_z_towel,
                 delta_z_color,
                 delta_z_water,
                 position_for_pick_up,
                 padding_pickup_z,
                 stroke_names_list):
        """
        POSITIVE delta values MORE UP
        """
        self.name = name
        self.stroke_names_list = stroke_names_list

        # Make deltas POSITIVE for LONGER brushes.
        # ADD deltas when using!
        self.height = height
        self.delta_height = height - reference_height
        self.delta_z_paint = delta_z_paint + self.delta_height
        self.delta_z_towel = delta_z_towel + self.delta_height
        self.delta_z_color = delta_z_color + self.delta_height
        self.delta_z_water = delta_z_water + self.delta_height
        self.position_for_pick_up = position_for_pick_up
        self.position_before_pick_up = copy.deepcopy(position_for_pick_up)
        self.position_before_pick_up[2] += padding_pickup_z
