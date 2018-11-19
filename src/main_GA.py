# Copyright(c) Eric Steinberger 2018

from src.config import BrushConfig
from src.learn_strokes.GA import GA
from main import get_args

ga = GA(args=get_args(name="StrokeGen"),
        brush_to_paint_with=BrushConfig.B6,
        brush_currently_on=BrushConfig.NOTHING_MOUNTED,
        stroke_name="test",
        n_different_strokes_per_generation=10,
        how_often_paint_each_stroke=7,
        start_stroke_length_mm=4,
        stroke_deepness_mm=1.6,
        build_fns=True)

while True:
    ga.next_generation()
