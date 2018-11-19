# Copyright(c) Eric Steinberger 2018

import numpy as np
import torch

from src.simulation.Environment import Environment

"""
Iterative Stroke Sampling is an expert-system brute force approach to combining a known library of strokes to
approximate a given image. ISS, as the name suggests, is an iterative process. in each iteration, the algorithm
tests N not-completely-random-but-pretty-random strokes against a manually defined loss function using manually defined
simulation parameters. It then (maybe - depending on a few things) adds that stroke to its command sequence.
In the end of the simulation procedure, the command sequence and the simulated painting are exported.

For a more detailed explanation of this part of the smART algorithm, either go through the code or visit
steinberger-ai.com/smART and scroll down to the section about ISS.
"""

from src.config import ColorConfig, BrushConfig


class IterativeStrokeSampler:
    def __init__(self,
                 args,
                 target_img,  # Pillow image
                 ):

        """
        All image tensors are color first.
        """
        self.args = args
        self.name = args.name

        self.device = torch.device("cuda:0" if args.use_gpu else "cpu")

        self.max_batch_size = args.max_batch_size_iss

        self.env = Environment(args=args,
                               target_img=target_img)
        self.env.reset()

        self.num_strokes_before_get_col_again = args.n_strokes_between_getting_color
        self.min_with_same_color = max(1, int(args.min_n_strokes_with_same_color_in_a_row))
        self.min_with_same_brush = max(1, int(args.min_n_strokes_with_same_brush_in_a_row))
        self.min_segmentation_width_px = args.min_segmentation_width_px
        self.error_threshold = args.error_threshold
        self.reconsider_threshold = args.reconsider_threshold

        self.n_segmentations = None

        # _________________________________________ tracking vars
        self.num_with_current_color = None
        self.num_with_current_brush = None
        self.num_with_brush = None
        self.num_strokes_since_last_getting_color = None
        self.num_strokes_since_last_brush_switch = None
        self.num_did_not_do_stroke_in_a_row = None
        self.reset()

    def reset(self):
        self.env.reset()

        self.n_segmentations = int(
            np.log(float(self.env.px_x_size + self.env.px_y_size) / self.min_segmentation_width_px))

        self.num_with_current_color = 0
        self.num_with_current_brush = 0
        self.num_with_brush = 0
        self.num_strokes_since_last_getting_color = 0
        self.num_strokes_since_last_brush_switch = 0
        self.num_did_not_do_stroke_in_a_row = 0

    def run(self,
            num_max_strokes_to_paint,
            num_stroke_samples_per_iter,
            save,
            ):

        print("Starting Iterative Stroke Sampling")
        self.reset()

        self.env.change_brush(new_brush=self._select_brush(edges=self.env.complete_edges))  # select starting brush
        self.env.change_color(new_color=self._select_color(edges=self.env.complete_edges))  # select starting color

        # ________________________________________ start iterative stroke sampling _____________________________________
        for iteration in range(num_max_strokes_to_paint):
            did_a_stroke = self._build_next_stroke(n_samples=num_stroke_samples_per_iter)

            # ____________________________________ periodically maybe change color _____________________________________
            if self.num_with_current_color >= self.min_with_same_color:
                new_color = self._select_color(edges=self.env.complete_edges)

                if new_color is not self.env.current_color:
                    self.env.change_color(new_color=new_color)
                    self.num_strokes_since_last_getting_color = 0
                    self.num_with_current_color = 0

            # ____________________________________ periodically maybe change brush _____________________________________
            if self.num_with_current_brush >= self.min_with_same_brush:
                new_brush = self._select_brush(edges=self.env.complete_edges)

                # pick new color for this brush. we swapped anyway, so might reselct color too...
                if new_brush is not self.env.current_brush:
                    self.env.change_brush(new_brush=new_brush)

                    new_color = self._select_color(edges=self.env.complete_edges)
                    self.env.change_color(new_color)
                    self.num_strokes_since_last_getting_color = 0
                    self.num_with_current_color = 0

                    self.num_strokes_since_last_brush_switch = 0
                    self.num_with_current_brush = 0

            # ____________________________________________ update counters _____________________________________________
            if did_a_stroke:
                self.num_strokes_since_last_getting_color += 1
                self.num_with_current_color += 1
                self.num_with_current_brush += 1
                self.num_did_not_do_stroke_in_a_row = 0

                # get same color again after some strokes
                if self.num_strokes_since_last_getting_color >= self.num_strokes_before_get_col_again:
                    self.env.get_color()
                    self.num_strokes_since_last_getting_color = 0

            else:
                # consider color change if no stroke was made.
                self.num_did_not_do_stroke_in_a_row += 1

                if self.num_did_not_do_stroke_in_a_row >= self.reconsider_threshold:
                    old_color = self.env.current_color

                    if np.random.random() > 0.8:
                        new_color = ColorConfig.ALL_COLORS_LIST[
                            np.random.randint(low=0, high=ColorConfig.N_COLORS)]
                    else:
                        new_color = self._select_color(self.env.complete_edges)

                    if new_color is not old_color:
                        self.env.change_color(new_color=new_color)
                        self.num_strokes_since_last_getting_color = 0
                        self.num_with_current_color = 0

                    self.num_did_not_do_stroke_in_a_row = 0

            if iteration % 20 == 19:
                print("ISS Iteration", iteration + 1, " :: Did", self.env.num_strokes_done, "strokes.")

        # _______________________________________ the end of the painting process ______________________________________
        return self.env.finished(save=save)

    def _build_next_stroke(self, n_samples):
        edges = torch.tensor(self.env.complete_edges, dtype=torch.int32, device=self.device)

        if np.random.random() > 0.5:
            for i in range(self.n_segmentations):
                self._pick_quarter_with_most_todo(edges=edges)

        segment_size_x = edges[1] - edges[0]
        segment_size_y = edges[3] - edges[2]

        x_space = segment_size_x
        y_space = segment_size_y
        x_off = edges[0]
        y_off = edges[2]

        # these ifs avoid painting over border of paper
        half_stroke_size = float(self.env.stroke_size_px) / 2.0
        if edges[0] == 0:
            x_off += half_stroke_size
            x_space -= half_stroke_size
        if edges[1] == self.env.px_x_size:
            x_space -= half_stroke_size
        if edges[2] == 0:
            y_off += half_stroke_size
            y_space -= half_stroke_size
        if edges[3] == self.env.px_y_size:
            y_space -= half_stroke_size

        # ________________________ Sample strokes for best color and pick best if it is not shit _______________________
        stroke_id, rotation_id, center_x, center_y, score = self._sample_brushes_and_get_best(n_samples=n_samples,
                                                                                              x_space=x_space,
                                                                                              y_space=y_space,
                                                                                              x_off=x_off,
                                                                                              y_off=y_off)
        if score >= self.error_threshold:
            return False

        # don't do stroke if other color same stroke would be better
        _s = self.env.strokes[self.env.current_brush][stroke_id][rotation_id].unsqueeze(0)
        _cx = torch.tensor([center_x], dtype=torch.long, device=self.device)
        _cy = torch.tensor([center_y], dtype=torch.long, device=self.device)
        for sample_color in ColorConfig.ALL_COLORS_LIST:
            if sample_color is not self.env.current_color:
                error_of_other_color = self.env.get_errors_of_batch(
                    strokes=_s,
                    centers_x=_cx,
                    centers_y=_cy,
                    colors_rgb=sample_color.rgb_torch_iss.to(device=self.device).unsqueeze(0)
                )
                if error_of_other_color < score:
                    return False

        # if the above did not disqualify the stroke, add it to the painting sequence!
        self.env.apply_stroke(stroke_id=stroke_id,
                              rotation_id=rotation_id,
                              center_x=center_x,
                              center_y=center_y)

        return True

    def _sample_brushes_and_get_best(self,
                                     n_samples,
                                     x_space,
                                     y_space,
                                     x_off,
                                     y_off,
                                     ):
        """
        creates a batch of random strokes, evaluates them and returns the best one.
        """

        stroke_ids = torch.randint(size=(n_samples,),
                                   low=0,
                                   high=self.env.strokes[self.env.current_brush].size(0),
                                   device=self.device, dtype=torch.long)

        rotation_ids = torch.randint(size=(n_samples,),
                                     low=0,
                                     high=self.args.n_stroke_rotations,
                                     device=self.device, dtype=torch.long)

        centers_x = torch.randint(size=(n_samples,), low=x_off, high=x_space + x_off, device=self.device,
                                  dtype=torch.long)
        centers_y = torch.randint(size=(n_samples,), low=y_off, high=y_space + y_off, device=self.device,
                                  dtype=torch.long)

        colors_rgb = self.env.current_color.rgb_torch_iss.to(device=self.device).unsqueeze(0).expand(n_samples, 3)

        errors = torch.zeros(size=(n_samples,), dtype=torch.float32, device=self.device)

        sub_batches, starts, ends = self.env.get_sub_batches(batched_tensors_dict={"stroke_ids": stroke_ids,
                                                                                   "rotation_ids": rotation_ids,
                                                                                   "centers_x": centers_x,
                                                                                   "centers_y": centers_y,
                                                                                   "colors_rgb": colors_rgb,
                                                                                   },
                                                             max_batch_size=self.max_batch_size)

        for i, batch in enumerate(sub_batches):
            errors[starts[i]:ends[i]] = self.env.get_errors_of_batch(
                strokes=self.env.strokes[self.env.current_brush][batch["stroke_ids"], batch["rotation_ids"]],
                centers_x=batch["centers_x"],
                centers_y=batch["centers_y"],
                colors_rgb=batch["colors_rgb"]
            )

        score, best_idx = errors.min(0)

        return stroke_ids[best_idx].item(), \
               rotation_ids[best_idx].item(), \
               centers_x[best_idx].item(), \
               centers_y[best_idx].item(), \
               score.item()

    def _pick_quarter_with_most_todo(self, edges):
        # if higher and wider than 1 px by checking if 2 or bigger
        if not (edges[1] - edges[0] < 2 or edges[3] - edges[2] < 2):
            tmp = torch.empty(4, 4, device=self.device, dtype=torch.int32)

            # [0][?] = bottomleft quarter
            # [1][?] = topleft quarter
            # [2][?] = bottomright quarter
            # [3][?] = topright quarter

            # [?][0] = left corner of the ? quarter
            # [?][1] = right corner of the ? quarter
            # [?][2] = bottom corner of the ? quarter
            # [?][3] = top corner of the ? quarter

            tmp[0][0] = edges[0]
            tmp[0][1] = ((edges[1] - edges[0]) / 2) + edges[0]
            tmp[0][2] = edges[2]
            tmp[0][3] = (edges[3] - edges[2]) / 2 + edges[2]

            tmp[1][0] = edges[0]
            tmp[1][1] = tmp[0][1]
            tmp[1][2] = tmp[0][3]
            tmp[1][3] = edges[3]

            tmp[2][0] = tmp[0][1]
            tmp[2][1] = edges[1]
            tmp[2][2] = edges[2]
            tmp[2][3] = tmp[0][3]

            tmp[3][0] = tmp[0][1]
            tmp[3][1] = edges[1]
            tmp[3][2] = tmp[0][3]
            tmp[3][3] = edges[3]

            # random
            if np.random.random() > 0.9:
                ind = np.random.randint(low=0, high=4)
                edges[0] = tmp[ind][0]
                edges[1] = tmp[ind][1]
                edges[2] = tmp[ind][2]
                edges[3] = tmp[ind][3]
                return

            # ______________________________ pick the quarter with the most to-do in it ________________________________

            winner = None
            top_score = -1.0  # score can never be under 0 so setting to -1 is sound

            for q in range(4):
                score = (self.env.todo_img[:, tmp[q][0]:tmp[q][1], tmp[q][2]:tmp[q][3]]).abs().sum()
                if score > top_score:
                    top_score = score
                    winner = q

            # ___________________________________ update edges with selected quarter ___________________________________
            edges[0] = tmp[winner][0]
            edges[1] = tmp[winner][1]
            edges[2] = tmp[winner][2]
            edges[3] = tmp[winner][3]

        else:
            raise ValueError("Can't quarter. Only 1 pixel left")

    def _select_brush(self, edges, n_samples_per_brush=100):
        """ selects brush object """

        # ______________________________________________ build samples _________________________________________________

        padding = self.env.stroke_size_px / 2.0
        min_x = int(edges[0] + padding)
        min_y = int(edges[2] + padding)
        max_x = int(edges[1] - edges[0] - 2 * padding) + min_x
        max_y = int(edges[3] - edges[2] - 2 * padding) + min_y

        centers_x = torch.randint(size=(n_samples_per_brush,), low=min_x, high=max_x, device=self.device,
                                  dtype=torch.long)
        centers_y = torch.randint(size=(n_samples_per_brush,), low=min_y, high=max_y, device=self.device,
                                  dtype=torch.long)

        colors_idxs = torch.randint(size=(n_samples_per_brush,), low=0, high=ColorConfig.N_COLORS,
                                    device=self.device,
                                    dtype=torch.long)

        colors_rgb = torch.stack([c.rgb_torch_iss.to(device=self.device) for c in
                                  ColorConfig.ALL_COLORS_LIST], dim=0)[colors_idxs]

        rotation_ids = torch.randint(size=(n_samples_per_brush,),
                                     low=0,
                                     high=self.args.n_stroke_rotations,
                                     device=self.device, dtype=torch.long)

        errors_per_brush = torch.empty(len(BrushConfig.ALL_PAINTING_BRUSHES_LIST))
        for i, b in enumerate(BrushConfig.ALL_PAINTING_BRUSHES_LIST):
            stroke_ids = torch.randint(size=(n_samples_per_brush,),
                                       low=0,
                                       high=self.env.strokes[b].size(0),
                                       device=self.device, dtype=torch.long)

            strokes = self.env.strokes[b][stroke_ids, rotation_ids]

            errors_per_brush[i] = self.env.get_errors_of_batch(strokes=strokes,
                                                               centers_x=centers_x,
                                                               centers_y=centers_y,
                                                               colors_rgb=colors_rgb).mean()

        # __________________________________________________ evaluate __________________________________________________
        best_score, best_b_idx = errors_per_brush.min(0)
        return BrushConfig.ALL_PAINTING_BRUSHES_LIST[best_b_idx]

    def _select_color(self, edges, n_samples_per_color=100):
        """ selects color object """

        padding = self.env.stroke_size_px / 2.0
        min_x = int(edges[0] + padding)
        min_y = int(edges[2] + padding)
        max_x = int(edges[1] - edges[0] - 2 * padding) + min_x
        max_y = int(edges[3] - edges[2] - 2 * padding) + min_y

        # first create a bunch of random strokes with the current brush

        stroke_ids = torch.randint(size=(n_samples_per_color,),
                                   low=0,
                                   high=self.env.strokes[self.env.current_brush].size(0),
                                   device=self.device, dtype=torch.long)

        rotation_ids = torch.randint(size=(n_samples_per_color,),
                                     low=0,
                                     high=self.args.n_stroke_rotations,
                                     device=self.device, dtype=torch.long)

        centers_x = torch.randint(size=(n_samples_per_color,), low=min_x, high=max_x, device=self.device,
                                  dtype=torch.long)
        centers_y = torch.randint(size=(n_samples_per_color,), low=min_y, high=max_y, device=self.device,
                                  dtype=torch.long)

        errors_per_color = torch.stack(
            tensors=tuple(
                [self.env.get_errors_of_batch(
                    strokes=self.env.strokes[self.env.current_brush][stroke_ids, rotation_ids],
                    centers_x=centers_x,
                    centers_y=centers_y,
                    colors_rgb=c.rgb_torch_iss.to(device=self.device).unsqueeze(0).expand(
                        n_samples_per_color, 3))
                    for c in ColorConfig.ALL_COLORS_LIST]),
            dim=0)

        # _______________________ don't count a stroke if another color with it would be better ________________________
        errors_per_color_adjusted = torch.zeros(ColorConfig.N_COLORS, dtype=torch.float32, device=self.device)
        mins, _ = errors_per_color.min(0)
        for c_idx in range(len(ColorConfig.ALL_COLORS_LIST)):
            e = errors_per_color[c_idx]
            err_of_samples_where_color_is_best = e[e == mins]
            if err_of_samples_where_color_is_best.size(
                    0) < n_samples_per_color / ColorConfig.N_COLORS / 10.0:
                errors_per_color_adjusted[c_idx] = 1e10
            else:
                errors_per_color_adjusted[c_idx] = err_of_samples_where_color_is_best.mean()

        best_score, best_c_idx = errors_per_color_adjusted.min(0)

        return ColorConfig.ALL_COLORS_LIST[best_c_idx]
