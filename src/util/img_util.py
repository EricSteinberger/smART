# Copyright(c) Eric Steinberger 2018

from PIL import Image


def resize_to_mm(image, px_per_mm, max_mm_x, max_mm_y):
    """
    If target and start ratio is not equaly, it scales up until 1 axis hits target. *KEEPING ORIGINAL RATIO*
    """
    x, y = image.size
    scale_ratio = min(max_mm_y / y, max_mm_x / x) * px_per_mm
    return image.resize((int(x * scale_ratio), int(y * scale_ratio)), Image.ANTIALIAS)
