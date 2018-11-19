# Copyright(c) Eric Steinberger 2018

import numpy as np
import torch


class Color:
    def __init__(self, name, rgb, position, color_id, code):
        self.name = name
        self.rgb = np.array(rgb) if not isinstance(rgb, np.ndarray) else rgb

        # ISS uses inverted colors
        self.rgb_torch_iss = 255 - torch.from_numpy(self.rgb).to(dtype=torch.float32)

        self.capital = name.title()
        self.position = position
        self.id = color_id
        self.code = code

